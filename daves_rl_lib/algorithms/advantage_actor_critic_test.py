import dataclasses

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import drivers
from daves_rl_lib import networks
from daves_rl_lib.algorithms import advantage_actor_critic
from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.environments import trivial_environment
from daves_rl_lib.internal import test_util
from daves_rl_lib.internal import util


class ActorCriticTests(test_util.TestCase):

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

    @parameterized.named_parameters([('_no_auxiliary', False),
                                     ('_auxiliary', True)])
    def test_step_preserves_shapes(self, keep_auxiliary):
        batch_size = 32
        steps_per_update = 2
        observation_size = 2
        action_size = 1
        observations = jnp.zeros([batch_size, observation_size])
        actions = jnp.zeros([batch_size, action_size])

        agent = advantage_actor_critic.A2CAgent(
            policy_net=networks.make_model(
                [1], obs_size=2, activate_final=tfp.distributions.Categorical),
            value_net=networks.make_model([1], obs_size=observation_size),
            policy_optimizer=optax.adam(1e-2),
            value_optimizer=optax.adam(1e-1),
            steps_per_update=steps_per_update,
            entropy_regularization=0.,
            keep_auxiliary=keep_auxiliary,
            discount_factor=1.)

        weights = agent.init_weights(seed=test_util.test_seed(),
                                     dummy_observation=observations[0, ...],
                                     dummy_action=actions[0, ...],
                                     batch_size=batch_size)
        self.assertAllEqualNested(
            jax.tree_util.tree_map(lambda x: x.shape, weights.steps_buffer),
            replay_buffer.ReplayBuffer(
                transitions=environment_lib.Transition(
                    observation=(batch_size, steps_per_update,
                                 observation_size),  # type: ignore
                    action=(batch_size, steps_per_update,
                            action_size),  # type: ignore
                    next_observation=(batch_size, steps_per_update,
                                      observation_size),  # type: ignore
                    reward=(batch_size, steps_per_update),  # type: ignore
                    done=(batch_size, steps_per_update)),  # type: ignore
                index=(batch_size,),  # type: ignore
                is_full=(batch_size,)))  # type: ignore

        batch_of_transitions = environment_lib.Transition(
            observation=observations,
            action=actions,
            next_observation=observations,
            reward=jnp.zeros([batch_size]),
            done=jnp.zeros([batch_size], dtype=bool))
        for _ in range(steps_per_update + 1):
            new_weights = agent.update(weights, batch_of_transitions)
            self.assertSameShapeNested(new_weights, weights)
            weights = new_weights

    @parameterized.named_parameters([('_no_batch', None), ('_batch', 8)])
    def test_estimates_value(self, batch_size):
        discount_factor = 0.5
        env = trivial_environment.OneStepEnvironment(
            discount_factor=discount_factor)
        states = env.reset(seed=test_util.test_seed(), batch_size=batch_size)
        initial_state_observation = states.observation
        if batch_size:
            initial_state_observation = initial_state_observation[0, ...]

        agent = advantage_actor_critic.A2CAgent(
            policy_net=networks.make_model(
                [1],
                obs_size=env.observation_size,
                activate_final=tfp.distributions.Categorical),
            value_net=networks.make_model([1], obs_size=env.observation_size),
            policy_optimizer=optax.adam(1e-1),
            value_optimizer=optax.adam(0.1, b1=0.5, b2=0.9),
            steps_per_update=7,
            entropy_regularization=0.,
            discount_factor=discount_factor)
        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=initial_state_observation,
            dummy_action=env.action_space.dummy_action(),
            batch_size=batch_size)

        step_fn = agent.update
        jitted_step_fn = jax.jit(step_fn)

        def do_step(state, weights, seed):
            state = env.reset_if_done(state)
            action = agent.action_dist(weights,
                                       state.observation).sample(seed=seed)
            next_state = env.step(state, action)
            return state, action, next_state

        do_step_fn = do_step
        if batch_size:
            do_step_fn = (
                lambda st, w, sd: jax.vmap(
                    do_step,
                    in_axes=(0, None, 0)  # type: ignore
                )(st, w, jax.random.split(sd, batch_size)))

        seed = test_util.test_seed()
        for i in range(200):
            seed, action_seed = jax.random.split(seed, 2)
            states, actions, next_states = do_step_fn(states, weights, seed)
            weights = jitted_step_fn(
                weights,
                environment_lib.Transition(
                    observation=states.observation,
                    action=actions,
                    next_observation=next_states.observation,
                    reward=next_states.reward,
                    done=next_states.done))
            states = next_states

        initial_state_value = agent.value_net.apply(
            weights.agent_weights.value_weights, initial_state_observation)
        np.testing.assert_allclose(initial_state_value, 1., atol=0.02)

    def test_learns_in_trivial_discrete_environment(self):
        batch_size = 128
        num_steps_inner = 8
        num_steps_outer = 20
        discount_factor = 0.9
        seed = test_util.test_seed()

        env = trivial_environment.DiscreteTargetEnvironment(
            size=2,
            dim=1,
            discount_factor=discount_factor,
            one_hot_features=True)
        agent = advantage_actor_critic.A2CAgent(
            policy_net=networks.make_model(
                [32, env._dim * 2],
                obs_size=env.observation_size,
                activate_final=networks.categorical_from_logits),
            value_net=networks.make_model([32, 32, 1],
                                          obs_size=env.observation_size),
            policy_optimizer=optax.adam(1e-1),
            value_optimizer=optax.adam(1e-2, b1=0.5),
            discount_factor=discount_factor,
            entropy_regularization=0.0,
            steps_per_update=num_steps_inner)

        weights = agent.init_weights(
            seed=seed,
            dummy_observation=env.reset(seed=seed).observation,
            dummy_action=env.action_space.dummy_action(),
            batch_size=batch_size)

        states = env.reset(seed=test_util.test_seed(), batch_size=batch_size)
        initial_state_obs = states.observation[0, ...]

        step_fn = jax.jit(drivers.jax_driver(env, agent))
        seed = test_util.test_seed()
        for _ in range(num_steps_outer * num_steps_inner):
            states, weights, seed = step_fn(states, weights, seed)
        # Value estimate for initial state.
        self.assertAllClose(agent.value_net.apply(
            weights.agent_weights.value_weights, initial_state_obs),
                            discount_factor,
                            atol=0.02)

        final_returns = np.array(states.episode_return)[states.done]
        final_lengths = np.array(states.num_steps)[states.done]
        self.assertAllClose(jnp.mean(final_returns), discount_factor, atol=0.02)
        self.assertAllClose(jnp.mean(final_lengths), 2., atol=0.02)

    def test_learns_in_trivial_continuous_environment(self):
        dim = 2
        batch_size = 512
        entropy_regularization = 0.0
        steps_per_update = 1
        discount_factor = 0.97

        env = trivial_environment.ContinuousEnvironmentStateless(
            dim=2, discount_factor=discount_factor)

        agent = advantage_actor_critic.A2CAgent(
            policy_net=networks.make_model(
                [dim],
                obs_size=env.observation_size,
                activate_final=(lambda x: tfp.distributions.
                                MultivariateNormalDiag(x, jnp.ones_like(x)))),
            value_net=networks.make_model([1], obs_size=env.observation_size),
            policy_optimizer=optax.adam(1e-1),
            value_optimizer=optax.adam(1e-1),
            discount_factor=discount_factor,
            entropy_regularization=entropy_regularization,
            steps_per_update=steps_per_update)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=env.reset(seed=test_util.test_seed()).observation,
            dummy_action=env.action_space.dummy_action(),
            batch_size=batch_size)

        states = env.reset(seed=test_util.test_seed(), batch_size=batch_size)

        step_fn = jax.jit(drivers.jax_driver(env, agent))
        seed = test_util.test_seed()
        for idx in range(5001):
            states, weights, seed = step_fn(states, weights, seed)
            if idx % 50 == 0:
                print(idx, jnp.mean(states.episode_return))

        self.assertGreater(jnp.mean(states.episode_return), -5.)


if __name__ == '__main__':
    absltest.main()