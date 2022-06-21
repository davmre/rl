import dataclasses

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import networks
from daves_rl_lib.algorithms import advantage_actor_critic
from daves_rl_lib.environments import trivial_environment
from daves_rl_lib.internal import test_util
from daves_rl_lib.internal import util


def trace_episode_length_and_reward(
        tq: advantage_actor_critic.A2CTraceableQuantities):
    end_of_episode = jnp.logical_and(tq.agent_states.done[1:],
                                     ~tq.agent_states.done[:-1])
    end_of_episode_idx = 1 + jnp.argmax(end_of_episode)
    return {
        'return':
            jnp.where(jnp.any(end_of_episode),
                      tq.agent_states.episode_return[end_of_episode_idx],
                      np.nan),
        'length':
            jnp.where(jnp.any(end_of_episode),
                      tq.agent_states.num_steps[end_of_episode_idx], -1),
    }


def setup_discrete_test_environment(
        seed,
        env_size=2,
        env_dim=1,
        batch_size=32,
        num_steps=16,
        discount_factor=0.97,
        entropy_regularization=0.01,
        value_learning_rate=1e-2,
        policy_learning_rate=1e-2,
        trace_fn: advantage_actor_critic.A2CTraceFn = lambda tq: []):

    env = trivial_environment.DiscreteTargetEnvironment(
        size=env_size,
        dim=env_dim,
        discount_factor=discount_factor,
        one_hot_features=True)
    policy_net = networks.make_model(
        [32, env._dim * 2],
        obs_size=env.observation_size,
        activate_final=networks.categorical_from_logits)
    value_net = networks.make_model([32, 32, 1], obs_size=env.observation_size)
    policy_optimizer = optax.adam(policy_learning_rate)
    value_optimizer = optax.adam(value_learning_rate)
    learner = advantage_actor_critic.initialize_learner(
        env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        batch_size=batch_size,
        seed=seed)

    step_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        num_steps=num_steps,
        entropy_regularization=entropy_regularization,
        trace_fn=trace_fn)

    return env, learner, step_fn, value_net


class ActorCriticTests(test_util.TestCase):

    def test_reward_to_go(self):
        rewards = [3., 7., -2.]
        rtg = advantage_actor_critic.rewards_to_go(jnp.array(rewards),
                                                   discount_factor=0.5)
        expected = jnp.array([3. + 0.5 * 7. - 0.25 * 2., 7 - 0.5 * 2., -2.])
        self.assertTrue(jnp.all(jnp.equal(rtg, expected)))

    def test_batch_policy_gradient(self):
        policy_seed, obs_seed, action_seed, advantage_seed = jax.random.split(
            jax.random.PRNGKey(0), num=4)
        num_steps = 17
        policy_net = networks.make_model(
            layer_sizes=[4],
            obs_size=2,
            activate_final=tfp.distributions.Categorical)
        policy_weights = policy_net.init(policy_seed)

        observations = jax.random.normal(obs_seed, shape=(num_steps, 2))
        action_dist = policy_net.apply(policy_weights, observations)
        actions = action_dist.sample(seed=action_seed)
        advantages = jax.random.normal(advantage_seed, shape=(num_steps,))

        policy_grad = advantage_actor_critic.batch_policy_gradient(
            policy_net,
            policy_weights,
            batch_obs=observations,
            batch_actions=actions,
            batch_advantages=advantages)

        # Compare to explicit calculation
        def score(obs, a):
            return jax.grad(lambda w: policy_net.apply(w, obs).log_prob(a))(
                policy_weights)

        expected_policy_grad = jax.tree_util.tree_map(
            lambda s: jnp.mean(util.batch_multiply(advantages, s), axis=0),
            jax.vmap(score)(observations, actions))

        self.assertAllCloseNested(policy_grad, expected_policy_grad, atol=1e-5)

    def test_step_preserves_shapes(self):
        batch_size = 32
        env, learner, step_fn, _ = setup_discrete_test_environment(
            seed=test_util.test_seed(), batch_size=batch_size)
        new_learner, _ = step_fn(learner, 0)
        self.assertSameShapeNested(learner, new_learner)

    def test_estimates_value(self):
        env = trivial_environment.OneStepEnvironment(discount_factor=0.5)
        policy_optimizer = optax.adam(1e-2)
        value_optimizer = optax.adam(1e-1)
        policy_net = networks.make_model(
            [1],
            obs_size=env.observation_size,
            activate_final=tfp.distributions.Categorical)
        value_net = networks.make_model([1], obs_size=env.observation_size)
        learner = advantage_actor_critic.initialize_learner(
            env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            batch_size=8,
            seed=test_util.test_seed())

        step_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            num_steps=7,
            entropy_regularization=0.)
        jitted_step_fn = jax.jit(step_fn)
        learner, _ = jax.lax.scan(jitted_step_fn, learner, jnp.arange(50))
        initial_state = env.reset(test_util.test_seed())
        initial_state_value = value_net.apply(learner.value_weights,
                                              initial_state.observation)
        np.testing.assert_allclose(initial_state_value, 1., atol=0.02)

    def test_extra_steps_are_no_ops(self):
        env = trivial_environment.ContinuousEnvironmentStateless(
            discount_factor=0.5)
        policy_net = networks.make_model(
            [1],
            obs_size=env.observation_size,
            activate_final=tfp.distributions.Categorical)
        value_net = networks.make_model([1], obs_size=env.observation_size)
        policy_optimizer = optax.adam(1e-2)
        value_optimizer = optax.adam(1e-1)
        learner = advantage_actor_critic.initialize_learner(
            env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            batch_size=8,
            seed=test_util.test_seed())

        step1_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            num_steps=1,
            entropy_regularization=0.)
        step7_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            num_steps=7,
            entropy_regularization=0.)
        learner1, _ = step1_fn(learner, 0)
        learner7, _ = step7_fn(learner, 0)

        learner1 = dataclasses.replace(learner1,
                                       agent_states=dataclasses.replace(
                                           learner1.agent_states,
                                           seed=learner7.agent_states.seed,
                                           reward=jnp.zeros_like(
                                               learner1.agent_states.reward)))

        self.assertAllCloseNested(learner1, learner7)

    def test_learns_in_trivial_discrete_environment(self):
        batch_size = 128
        num_steps_inner = 8
        num_steps_outer = 101
        discount_factor = 0.9

        env, learner, step_fn, value_net = setup_discrete_test_environment(
            seed=test_util.test_seed(),
            batch_size=batch_size,
            discount_factor=discount_factor,
            num_steps=num_steps_inner,
            entropy_regularization=0.0,
            policy_learning_rate=1e-1,
            trace_fn=trace_episode_length_and_reward)
        initial_state_obs = learner.agent_states.observation[0, ...]
        learner, diagnostics = jax.lax.scan(jax.jit(step_fn), learner,
                                            jnp.arange(num_steps_outer))

        # Value estimate for initial state.
        self.assertAllClose(value_net.apply(learner.value_weights,
                                            initial_state_obs),
                            discount_factor,
                            atol=0.02)

        self.assertEqual(diagnostics['return'].shape,
                         (num_steps_outer, batch_size))
        self.assertAllClose(jnp.mean(diagnostics['return'][-1, ...]),
                            discount_factor,
                            atol=0.02)
        self.assertAllClose(jnp.mean(diagnostics['length'][-1, ...]),
                            2.,
                            atol=0.02)

    def test_learns_in_trivial_continuous_environment(self):
        dim = 2

        env = trivial_environment.ContinuousEnvironmentStateless(dim=2)
        policy_net = networks.make_model(
            [dim],
            obs_size=env.observation_size,
            activate_final=(lambda x: tfp.distributions.MultivariateNormalDiag(
                x, jnp.ones_like(x))))
        value_net = networks.make_model([1], obs_size=env.observation_size)
        policy_optimizer = optax.adam(1e-1)
        value_optimizer = optax.adam(1e-1)
        learner = advantage_actor_critic.initialize_learner(
            env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            batch_size=512,
            seed=jax.random.PRNGKey(0))

        step_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            num_steps=1,
            entropy_regularization=0.0,
            trace_fn=trace_episode_length_and_reward)

        learner, diagnostics = jax.lax.scan(jax.jit(step_fn), learner,
                                            jnp.arange(5001))

        self.assertGreater(jnp.mean(diagnostics['return'][-1, ...]), -5.)


if __name__ == '__main__':
    absltest.main()