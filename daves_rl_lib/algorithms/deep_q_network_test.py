import dataclasses
from statistics import mean
from typing import Optional

from absl.testing import parameterized
import gym

import jax
from jax import numpy as jnp
import numpy as np

import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import drivers
from daves_rl_lib import networks
from daves_rl_lib.algorithms import deep_q_network
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.environments import gym_environment
from daves_rl_lib.environments import trivial_environment
from daves_rl_lib.internal import test_util


class DQNTests(test_util.TestCase):

    @parameterized.named_parameters([('_no_batch', np.asarray([1., 2.]), 4),
                                     ('_batch',
                                      np.asarray([[1., 2.], [3., 4.],
                                                  [5., 6.]]), 4),
                                     ('_one_action', np.array([1., 2.]), 1)])
    def test_epsilon_greedy(self, obs, num_actions):
        epsilon = 1e-1
        batch_shape = obs.shape[:-1]
        net = networks.make_model([num_actions], obs_size=obs.shape[-1])
        agent = deep_q_network.DQNAgent(qvalue_net=net,
                                        qvalue_optimizer=optax.adam(0.1),
                                        replay_buffer_size=0,
                                        gradient_batch_size=8,
                                        epsilon=epsilon,
                                        target_weights_decay=0.9)

        weights = agent.init_weights(test_util.test_seed())
        qvalues = net.apply(weights.qvalue_weights, obs)
        self.assertEqual(qvalues.shape, batch_shape + (num_actions,))
        best_actions = jnp.argmax(qvalues, axis=-1)

        action_dist = agent.action_dist(weights, obs)
        self.assertEqual(action_dist.batch_shape, batch_shape)
        self.assertEqual(action_dist.event_shape, ())
        probs = action_dist[..., jnp.newaxis].prob(jnp.arange(num_actions))
        probs_of_best_actions = jnp.take_along_axis(probs,
                                                    best_actions[..., None],
                                                    axis=-1)[..., 0]
        if num_actions > 1:
            self.assertAllClose(
                probs_of_best_actions,
                jnp.ones(batch_shape) * (1 - epsilon) + epsilon / num_actions)
        else:
            self.assertAllClose(probs_of_best_actions, 1.)

    def test_transitions_are_collected_in_buffer(self):
        env = trivial_environment.OneStepEnvironment(discount_factor=0.9)
        batch_size = 6
        buffer_size = 8
        states = env.reset(seed=test_util.test_seed(), batch_size=batch_size)
        initial_state_obs = states.observation[0, ...]

        agent = deep_q_network.DQNAgent(
            qvalue_net=networks.make_model(
                [env.action_space.num_actions],  # type: ignore
                obs_size=env.observation_size),
            qvalue_optimizer=optax.adam(0.1),
            replay_buffer_size=buffer_size,
            gradient_batch_size=8,
            epsilon=0.,
            target_weights_decay=0.9,
            discount_factor=env.discount_factor)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=env.reset(test_util.test_seed()).observation,
            dummy_action=env.action_space.dummy_action())

        actions = env.action_space.dummy_action(batch_size=batch_size)
        next_states = jax.vmap(env.step)(states, actions)
        weights = agent.update(weights,
                               transition=environment_lib.Transition(
                                   observation=states.observation,
                                   action=actions,
                                   next_observation=next_states.observation,
                                   reward=next_states.reward,
                                   done=next_states.done))

        self.assertEqual(weights.replay_buffer.index, batch_size)
        self.assertFalse(weights.replay_buffer.is_full)

        states = jax.vmap(env.reset_if_done)(next_states)
        next_states = jax.vmap(env.step)(states, actions)
        weights = agent.update(weights,
                               transition=environment_lib.Transition(
                                   observation=states.observation,
                                   action=actions,
                                   next_observation=next_states.observation,
                                   reward=next_states.reward,
                                   done=next_states.done))
        self.assertEqual(weights.replay_buffer.index,
                         (batch_size * 2) % buffer_size)
        self.assertTrue(weights.replay_buffer.is_full)
        self.assertAllEqual(weights.replay_buffer.transitions.observation,
                            jnp.array([initial_state_obs] * buffer_size))
        self.assertAllEqual(weights.replay_buffer.transitions.done, True)

    def test_update_qvalue_network(self):
        discount_factor = 0.8
        env = trivial_environment.OneStepEnvironment(
            discount_factor=discount_factor)
        buffer_size = 8
        states = env.reset(seed=test_util.test_seed())

        agent = deep_q_network.DQNAgent(
            qvalue_net=networks.make_model(
                [env.action_space.num_actions],  # type: ignore
                obs_size=env.observation_size),
            qvalue_optimizer=optax.adam(0.5),
            replay_buffer_size=buffer_size,
            gradient_batch_size=4,
            epsilon=0.,
            target_weights_decay=0.5,
            discount_factor=discount_factor)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=env.reset(test_util.test_seed()).observation,
            dummy_action=env.action_space.dummy_action())

        # Transitions with rewards 0.5 and 1.0 of equal probability, so
        # that the minimum possible TD error is 0.25.
        reward = jnp.array([1.] * (buffer_size // 2) + [0.5] *
                           (buffer_size // 2))
        transitions = environment_lib.Transition(
            observation=jnp.zeros([buffer_size, 1]),
            action=jnp.zeros([buffer_size], dtype=int),
            next_observation=jnp.zeros([buffer_size, 1]),
            done=jnp.zeros([buffer_size], dtype=bool),
            reward=reward)
        weights = dataclasses.replace(
            weights,
            replay_buffer=weights.replay_buffer.with_transitions(transitions),
            qvalue_target_weights=jax.tree_util.tree_map(
                jnp.zeros_like, weights.qvalue_target_weights))

        mean_abs_td_error = jax.jit(lambda w: jnp.mean(
            jnp.abs(
                deep_q_network.qvalues_and_td_error(
                    transition=w.replay_buffer.transitions,
                    qvalue_net=agent.qvalue_net,
                    qvalue_weights=w.qvalue_weights,
                    qvalue_target_weights=w.qvalue_target_weights,
                    discount_factor=discount_factor)[-1])))

        td_error = mean_abs_td_error(weights)
        updated_weights = agent.update(weights, transitions)
        updated_td_error = mean_abs_td_error(updated_weights)

        # Check that the TD error is decreasing and converges to the expected
        # minimum.
        for i in range(3):
            self.assertLessEqual(updated_td_error, td_error)
            weights, td_error = updated_weights, updated_td_error
            updated_weights = agent.update(weights, transitions)
            updated_td_error = mean_abs_td_error(updated_weights)

        self.assertAllClose(updated_td_error, 0.25, atol=1e-5)

    def test_learns_in_trivial_discrete_environment(self):

        env = trivial_environment.DiscreteTargetEnvironment(
            size=2, dim=1, discount_factor=0.6, one_hot_features=True)
        buffer_size = 64
        epsilon = 0.5
        states = env.reset(seed=test_util.test_seed(), batch_size=8)
        initial_state_obs = states.observation[0, ...]

        agent = deep_q_network.DQNAgent(
            qvalue_net=networks.make_model(
                [env.action_space.num_actions],  # type: ignore
                obs_size=env.observation_size),
            qvalue_optimizer=optax.adam(0.1),
            replay_buffer_size=buffer_size,
            gradient_batch_size=8,
            epsilon=epsilon,
            target_weights_decay=0.9,
            discount_factor=env.discount_factor)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=env.reset(test_util.test_seed()).observation,
            dummy_action=env.action_space.dummy_action())

        step_fn = jax.jit(drivers.jax_driver(env=env, agent=agent))

        seed = test_util.test_seed()
        done = []
        returns = []
        for _ in range(256):
            states, weights, seed = step_fn(states, weights, seed)
            done.append(states.done)
            returns.append(states.episode_return)
        done, returns = jnp.array(done), jnp.array(returns)

        num_episodes = jnp.sum(done)
        final_returns = jnp.where(done, returns, 0.)
        mean_return = jnp.sum(final_returns) / num_episodes

        self.assertLess(mean_return, env.discount_factor)
        self.assertGreater(mean_return, (1 - epsilon)**2 * env.discount_factor)

        self.assertAllClose(agent.qvalue_net.apply(weights.qvalue_weights,
                                                   initial_state_obs),
                            [env.discount_factor**3, env.discount_factor],
                            atol=0.03)

    def test_gym_cartpole(self):
        buffer_size = 64
        epsilon = 0.1
        discount_factor = 0.6
        env = gym_environment.GymEnvironment(gym.make("CartPole-v1"),
                                             discount_factor=discount_factor)
        state = env.reset(seed=test_util.test_seed())
        initial_state_obs = state.observation

        agent = deep_q_network.DQNAgent(
            qvalue_net=networks.make_model(
                [env.action_space.num_actions],  # type: ignore
                obs_size=env.observation_size),
            qvalue_optimizer=optax.adam(0.1),
            replay_buffer_size=buffer_size,
            gradient_batch_size=8,
            epsilon=epsilon,
            target_weights_decay=0.9,
            discount_factor=env.discount_factor)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=env.reset(test_util.test_seed()).observation,
            dummy_action=env.action_space.dummy_action())

        step_fn = drivers.stateful_driver(env=env, agent=agent)

        seed = test_util.test_seed()
        for _ in range(256):
            state, weights, seed = step_fn(state, weights, seed)