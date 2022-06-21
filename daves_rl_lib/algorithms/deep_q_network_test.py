import dataclasses
from typing import Optional
from statistics import mean

import gym

import numpy as np
import jax
from jax import numpy as jnp
import optax

from daves_rl_lib.algorithms import deep_q_network
from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import trivial_environment
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.algorithms import exploration_lib
from daves_rl_lib import networks
from daves_rl_lib.internal import test_util

from tensorflow_probability.substrates import jax as tfp


class DQNTests(test_util.TestCase):

    def _setup_one_step_learner(self,
                                env,
                                buffer_size,
                                batch_size: Optional[int] = 1,
                                learning_rate=1.0):
        qvalue_net = networks.make_model([env.action_space.num_actions],
                                         obs_size=env.observation_size)
        qvalue_optimizer = optax.adam(learning_rate)
        learner = deep_q_network.initialize_learner(
            env=env,
            qvalue_net=qvalue_net,
            qvalue_optimizer=qvalue_optimizer,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=test_util.test_seed())
        return qvalue_net, qvalue_optimizer, learner

    def test_collect_and_buffer_transitions(self):
        env = trivial_environment.OneStepEnvironment(discount_factor=0.9)
        batch_size = 6
        buffer_size = 8
        qvalue_net, _, learner = self._setup_one_step_learner(
            env, buffer_size=buffer_size, batch_size=batch_size)
        learner = deep_q_network.collect_and_buffer_jax_transitions(
            env=env, learner=learner, qvalue_net=qvalue_net, epsilon=0.)
        self.assertEqual(learner.replay_buffer.index, batch_size)
        self.assertFalse(learner.replay_buffer.is_full)

        # Check that environment gets reset.
        learner = deep_q_network.collect_and_buffer_jax_transitions(
            env=env, learner=learner, qvalue_net=qvalue_net, epsilon=0.)
        self.assertEqual(learner.replay_buffer.index,
                         (batch_size * 2) % buffer_size)
        self.assertTrue(learner.replay_buffer.is_full)
        self.assertAllEqual(learner.replay_buffer.transitions.state.done, False)
        self.assertAllEqual(learner.replay_buffer.transitions.next_state.done,
                            True)

    def test_update_qvalue_network(self):
        env = trivial_environment.OneStepEnvironment()
        buffer_size = 8
        qvalue_net, qvalue_optimizer, learner = self._setup_one_step_learner(
            env, buffer_size=buffer_size, learning_rate=0.5)
        # Transitions with rewards 0.5 and 1.0 of equal probability, so
        # that the minimum possible TD error is 0.25.
        dummy_states = environment_lib.State(
            observation=jnp.zeros([buffer_size, 1]),
            done=jnp.zeros([buffer_size], dtype=bool),
            reward=jnp.zeros([buffer_size]),
            num_steps=jnp.zeros([buffer_size], dtype=jnp.int32),
            episode_return=jnp.zeros([buffer_size]),
            seed=test_util.test_seed())
        reward = jnp.array([1.] * (buffer_size // 2) + [0.5] *
                           (buffer_size // 2))
        transitions = replay_buffer.Transition(
            state=dummy_states,
            action=jnp.zeros([buffer_size], dtype=int),
            next_state=dataclasses.replace(dummy_states,
                                           reward=reward,
                                           episode_return=reward,
                                           num_steps=jnp.ones_like(
                                               dummy_states.num_steps)),
            td_error=jnp.zeros([buffer_size]))
        learner = dataclasses.replace(
            learner,
            replay_buffer=learner.replay_buffer.with_transitions(transitions),
            qvalue_target_weights=jax.tree_util.tree_map(
                jnp.zeros_like, learner.qvalue_target_weights))

        # Compile the network update and TD error computation.
        update_network = jax.jit(lambda l: deep_q_network.update_qvalue_network(
            l,
            qvalue_net,
            qvalue_optimizer,
            gradient_batch_size=4,
            target_weights_decay=0.5,
            discount_factor=env.discount_factor))
        mean_abs_td_error = jax.jit(lambda l: jnp.mean(
            jnp.abs(
                deep_q_network.qvalues_and_td_error(
                    state=l.replay_buffer.transitions.state,
                    action=l.replay_buffer.transitions.action,
                    next_state=l.replay_buffer.transitions.next_state,
                    qvalue_net=qvalue_net,
                    qvalue_weights=l.qvalue_weights,
                    qvalue_target_weights=l.qvalue_target_weights,
                    discount_factor=0.8)[-1])))

        td_error = mean_abs_td_error(learner)
        updated_learner = update_network(learner)
        updated_td_error = mean_abs_td_error(updated_learner)

        # Sanity-check that updating the network from replays doesn't affect
        # the environment state.
        self.assertAllEqualNested(updated_learner.agent_states,
                                  learner.agent_states)

        # Check that the TD error is decreasing and converges to the expected
        # minimum.
        for i in range(3):
            self.assertLessEqual(updated_td_error, td_error)
            learner, td_error = updated_learner, updated_td_error
            updated_learner = update_network(learner)
            updated_td_error = mean_abs_td_error(updated_learner)

        self.assertAllClose(updated_td_error, 0.25, atol=1e-5)

    def test_learns_in_trivial_discrete_environment(self):

        env = trivial_environment.DiscreteTargetEnvironment(
            size=2, dim=1, discount_factor=0.6, one_hot_features=True)
        buffer_size = 64
        epsilon = 0.5
        qvalue_net, qvalue_optimizer, learner = self._setup_one_step_learner(
            env, buffer_size=buffer_size, batch_size=8, learning_rate=0.1)
        initial_state_obs = learner.agent_states.observation[0, ...]

        def trace(dql):
            return {
                'initial_qs':
                    qvalue_net.apply(dql.qvalue_weights, initial_state_obs),
                'return':
                    dql.agent_states.episode_return,
                'done':
                    dql.agent_states.done,
            }

        def loop_body(learner, _):
            learner = deep_q_network.deep_q_update_step(
                learner,
                env=env,
                qvalue_net=qvalue_net,
                qvalue_optimizer=qvalue_optimizer,
                gradient_batch_size=8,
                target_weights_decay=0.9,
                epsilon=epsilon)
            return learner, trace(learner)

        learner, diagnostics = jax.lax.scan(jax.jit(loop_body),
                                            init=learner,
                                            xs=jnp.arange(256))

        num_episodes = jnp.sum(diagnostics['done'])
        final_returns = jnp.where(diagnostics['done'], diagnostics['return'],
                                  0.)
        mean_return = jnp.sum(final_returns) / num_episodes

        self.assertLess(mean_return, env.discount_factor)
        self.assertGreater(mean_return, (1 - epsilon)**2 * env.discount_factor)

        self.assertAllClose(diagnostics['initial_qs'][-1, :],
                            [env.discount_factor**3, env.discount_factor],
                            atol=0.03)

    def test_gym_cartpole(self):
        buffer_size = 64
        epsilon = 0.1
        discount_factor = 0.6
        env = environment_lib.GymEnvironment(gym.make("CartPole-v1"),
                                             discount_factor=discount_factor)
        qvalue_net, qvalue_optimizer, learner = self._setup_one_step_learner(
            env, buffer_size=buffer_size, batch_size=None, learning_rate=0.1)
        step_learner = deep_q_network.compile_deep_q_update_step_stateful(
            env=env,
            qvalue_net=qvalue_net,
            qvalue_optimizer=qvalue_optimizer,
            gradient_batch_size=8,
            target_weights_decay=0.9,
            epsilon=epsilon)
        initial_obs = learner.agent_states.observation
        for step in range(256):
            learner = step_learner(learner)
        # Value estimate for initial state should be close to optimal.
        self.assertAllClose(np.max(
            qvalue_net.apply(learner.qvalue_weights, initial_obs)),
                            1 / (1 - discount_factor),
                            rtol=0.15)
        # Average per-step reward should be close to 1.
        self.assertAllClose(np.mean(
            learner.replay_buffer.valid_transitions().next_state.reward),
                            1.0,
                            atol=0.05)
