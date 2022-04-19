import functools
import dataclasses
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
from jax import numpy as jnp
from jax._src import test_util as jtu

import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import networks
from daves_rl_lib import train
from daves_rl_lib import util

from daves_rl_lib.brax_stuff import advantage_actor_critic
from daves_rl_lib.brax_stuff import trivial_environment
from daves_rl_lib.util import batch_multiply


def diagnose(tq):
    return {
        'mean entropy': tq.policy_entropy,
        #'entropy grad norm':
        #    jax.tree_util.tree_map(jnp.linalg.norm, tq.entropy_grad),
        #'policy grad norm':
        #    jax.tree_util.tree_map(jnp.linalg.norm, tq.policy_grad),
        #'value grad norm':
        #    jax.tree_util.tree_map(jnp.linalg.norm, tq.values_grad),''
        'mean advantage': jnp.mean(tq.advantage),
        'mean state value': jnp.mean(tq.state_values),
        'returns': jnp.mean(tq.returns)
    }


def setup_discrete_test_environment(seed,
                                    env_size=2,
                                    env_dim=1,
                                    batch_size=32,
                                    num_steps=16,
                                    discount_factor=0.97,
                                    entropy_regularization=0.01,
                                    value_learning_rate=1e-2,
                                    policy_learning_rate=1e-2,
                                    trace_fn=lambda tq: []):

    env = trivial_environment.DiscreteTargetEnvironment(size=env_size,
                                                        dim=env_dim)
    policy_optimizer = optax.adam(policy_learning_rate)
    value_optimizer = optax.adam(value_learning_rate)
    policy_net, value_net, learner = advantage_actor_critic.initialize_learner(
        env,
        policy_layer_sizes=[32, env._dim * 2],
        value_layer_sizes=[32, 32, 1],
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_activate_final=tfp.distributions.Categorical,
        batch_size=batch_size,
        seed=seed)

    step_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        discount_factor=discount_factor,
        num_steps=num_steps,
        entropy_regularization=entropy_regularization,
        trace_fn=trace_fn)

    return env, learner, step_fn, value_net


class ActorCriticTests(parameterized.TestCase):

    def assertSameShape(self, a, b):
        self.assertEqual(a.shape, b.shape)

    def assertSameShapeNested(self, a, b):
        jax.tree_util.tree_map(self.assertSameShape, a, b)

    def assertAllClose(self, a, b, **kwargs):
        np.testing.assert_allclose(np.float32(a), np.float32(b), **kwargs)

    def assertAllCloseNested(self, a, b, **kwargs):
        jax.tree_util.tree_map(lambda x, y: self.assertAllClose(x, y, **kwargs),
                               a, b)

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
            seed=jax.random.PRNGKey(0), batch_size=batch_size)
        new_learner, diagnostics = step_fn(learner, 0)
        self.assertSameShapeNested(learner, new_learner)

    def test_tracks_length_and_value(self):
        discount_factor = 0.9
        env_size = 3
        env = trivial_environment.DiscreteTargetEnvironment(size=env_size,
                                                            dim=1)
        policy_net = networks.FeedForwardModel(
            init=lambda _: [],
            apply=lambda *_: tfp.distributions.Categorical(logits=[0., 100.]))
        take_action = jax.jit(
            advantage_actor_critic.make_take_action(
                env,
                policy_net=policy_net,
                policy_weights=[],
                discount_factor=discount_factor))
        agent_state = advantage_actor_critic.AgentState(
            state=env.reset(),
            step=0,
            accumulated_reward=0.,
            last_episode_length=0,
            last_episode_return=0.,
            seed=jax.random.PRNGKey(0))
        for step in range(env_size - 1):
            agent_state, _ = take_action(agent_state, 0)
            self.assertEqual(agent_state.step, step + 1)
            self.assertEqual(agent_state.last_episode_length, 0)
            self.assertEqual(agent_state.accumulated_reward, 0.)
            self.assertEqual(agent_state.last_episode_return, 0.)
        # Step to the terminal state.
        agent_state, _ = take_action(agent_state, 0)
        expected_return = discount_factor**(env_size - 1)
        self.assertTrue(agent_state.state.done)
        self.assertEqual(agent_state.step, env_size)
        self.assertEqual(agent_state.last_episode_length, env_size)
        np.testing.assert_allclose(agent_state.accumulated_reward,
                                   expected_return)
        np.testing.assert_allclose(agent_state.last_episode_return,
                                   expected_return)

        # Attempt to step past the terminal state.
        agent_state, _ = take_action(agent_state, 0)
        self.assertEqual(agent_state.step, env_size)
        self.assertEqual(agent_state.last_episode_length, env_size)
        np.testing.assert_allclose(agent_state.accumulated_reward,
                                   expected_return)
        np.testing.assert_allclose(agent_state.last_episode_return,
                                   expected_return)

        # Reset the state and advance one step.
        agent_state, _ = take_action(
            advantage_actor_critic.maybe_reset_agent_state(env, agent_state), 0)
        self.assertEqual(agent_state.step, 1)
        self.assertEqual(agent_state.last_episode_length, env_size)
        np.testing.assert_allclose(agent_state.accumulated_reward, 0.)
        np.testing.assert_allclose(agent_state.last_episode_return,
                                   expected_return)

    def test_estimates_value(self):
        env = trivial_environment.OneStepEnvironment()
        policy_optimizer = optax.adam(1e-2)
        value_optimizer = optax.adam(1e-1)
        policy_net, value_net, learner = advantage_actor_critic.initialize_learner(
            env,
            policy_layer_sizes=[1],
            value_layer_sizes=[1],
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            policy_activate_final=tfp.distributions.Categorical,
            batch_size=8,
            seed=jax.random.PRNGKey(0))

        step_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            discount_factor=0.5,
            num_steps=7,
            entropy_regularization=0.)
        jitted_step_fn = jax.jit(step_fn)
        learner, _ = jax.lax.scan(jitted_step_fn, learner, jnp.arange(50))
        initial_state_value = value_net.apply(
            learner.value_weights,
            env.reset(jax.random.PRNGKey(0)).obs)
        np.testing.assert_allclose(initial_state_value, 1., atol=0.02)

    def test_extra_steps_are_no_ops(self):
        env = trivial_environment.ContinuousEnvironmentStateless()
        policy_optimizer = optax.adam(1e-2)
        value_optimizer = optax.adam(1e-1)
        policy_net, value_net, learner = advantage_actor_critic.initialize_learner(
            env,
            policy_layer_sizes=[1],
            value_layer_sizes=[1],
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            policy_activate_final=tfp.distributions.Categorical,
            batch_size=8,
            seed=jax.random.PRNGKey(0))

        step1_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            discount_factor=0.5,
            num_steps=1,
            entropy_regularization=0.)
        step7_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            discount_factor=0.5,
            num_steps=7,
            entropy_regularization=0.)
        learner1, _ = step1_fn(learner, 0)
        learner7, _ = step7_fn(learner, 0)
        self.assertAllCloseNested(learner1, learner7)

    def test_learns_in_trivial_discrete_environment(self):
        batch_size = 128
        discount_factor = 0.9

        def trace(tq):
            return {
                'entropy':
                    tq.policy_entropy,
                'initial_value':
                    value_net.apply(tq.value_weights, initial_state_obs),
                'return':
                    tq.agent_states.last_episode_return[-1],
                'length':
                    tq.agent_states.last_episode_length[-1],
            }

        env, learner, step_fn, value_net = setup_discrete_test_environment(
            seed=jax.random.PRNGKey(0),
            batch_size=batch_size,
            discount_factor=discount_factor,
            entropy_regularization=0.0,
            policy_learning_rate=1e-1,
            trace_fn=trace)
        initial_state_obs = learner.agent_states.state.obs[0, ...]
        learner, diagnostics = jax.lax.scan(jax.jit(step_fn), learner,
                                            jnp.arange(101))
        np.testing.assert_allclose(jnp.mean(diagnostics['return'][-1, ...]),
                                   discount_factor,
                                   atol=0.02)
        np.testing.assert_allclose(jnp.mean(diagnostics['initial_value'][-1,
                                                                         ...]),
                                   discount_factor,
                                   atol=0.02)
        np.testing.assert_allclose(jnp.mean(diagnostics['length'][-1, ...]),
                                   2.,
                                   atol=0.02)
        np.testing.assert_allclose(jnp.mean(diagnostics['entropy'][-1, ...]),
                                   0.,
                                   atol=0.02)

    def test_learns_in_trivial_continuous_environment(self):
        dim = 2

        env = trivial_environment.ContinuousEnvironmentStateless(dim=2)
        policy_optimizer = optax.adam(1e-2)
        value_optimizer = optax.adam(1e-2)
        policy_net, value_net, learner = advantage_actor_critic.initialize_learner(
            env,
            policy_layer_sizes=[dim],
            value_layer_sizes=[1],
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            policy_activate_final=lambda x: tfp.distributions.
            MultivariateNormalDiag(x, jnp.ones_like(x)),
            batch_size=512,
            seed=jax.random.PRNGKey(0))

        step_fn = advantage_actor_critic.make_advantage_actor_critic_batch_step(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            num_steps=1,
            entropy_regularization=0.0)

        learner, diagnostics = jax.lax.scan(jax.jit(step_fn), learner,
                                            jnp.arange(5001))
        self.assertGreater(jnp.mean(learner.agent_states.last_episode_return),
                           -5.)


if __name__ == '__main__':
    absltest.main()