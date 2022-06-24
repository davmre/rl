from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import test_util


def dummy_transitions(num_transitions, seed, action_is_vector=True):
    observations = jax.random.randint(seed,
                                      shape=[num_transitions],
                                      minval=0,
                                      maxval=1000)
    actions = jax.random.randint(
        seed,
        shape=[num_transitions, 1] if action_is_vector else [num_transitions],
        minval=-10,
        maxval=10)
    return environment_lib.Transition(
        observation=observations,
        action=actions,
        next_observation=observations +
        (actions[..., 0] if action_is_vector else actions),
        done=jnp.zeros([num_transitions], dtype=bool),
        reward=jnp.zeros([num_transitions]))


class ReplayBufferTests(test_util.TestCase):

    @parameterized.named_parameters([('_scalar_action', False),
                                     ('_vector_action', True)])
    def test_initialize_and_fill_buffer(self, action_is_vector):
        buffer_size = 32
        buffer = replay_buffer.ReplayBuffer.initialize_empty(
            size=buffer_size,
            observation=jnp.array(1),
            action=jnp.zeros((1,) if action_is_vector else ()))
        self.assertEqual(buffer.size, buffer_size)
        self.assertEqual(buffer.transitions.observation.shape, (buffer_size,))
        self.assertEqual(buffer.index, 0)
        self.assertFalse(buffer.is_full)

        transitions = dummy_transitions(num_transitions=buffer_size,
                                        seed=test_util.test_seed(),
                                        action_is_vector=action_is_vector)
        full_buffer = buffer.with_transitions(transitions)
        self.assertAllEqualNested(full_buffer.transitions, transitions)
        self.assertEqual(full_buffer.index, 0)
        self.assertTrue(full_buffer.is_full)

    @parameterized.named_parameters([
        ('', lambda b, t: b.with_transitions(t)),
        ('_jit', lambda b, t: jax.jit(b.with_transitions)(t)),
    ])
    def test_wrap_around(self, with_transitions):
        buffer_size = 32
        buffer = replay_buffer.ReplayBuffer.initialize_empty(
            size=buffer_size, observation=jnp.array(1), action=jnp.zeros([]))
        transitions = dummy_transitions(num_transitions=buffer_size - 5,
                                        seed=test_util.test_seed(),
                                        action_is_vector=False)
        buffer = with_transitions(buffer, transitions)
        self.assertEqual(buffer.index, buffer_size - 5)
        self.assertFalse(buffer.is_full)
        self.assertAllEqualNested(transitions, buffer.valid_transitions())

        buffer = with_transitions(buffer, transitions)
        self.assertEqual(buffer.index, buffer_size - 10)
        self.assertTrue(buffer.is_full)
        # First five transitions are added to the end of the array.
        self.assertAllEqual(transitions.observation[:5],
                            buffer.transitions.observation[-5:])
        # Remaining transitions wrap around to the beginning.
        self.assertAllEqual(transitions.observation[5:],
                            buffer.transitions.observation[:buffer_size - 10])
        # The wraparound doesn't fully overwrite the previous entries, so they
        # should still be there.
        self.assertAllEqual(transitions.observation[-5:],
                            buffer.transitions.observation[-10:-5])

    def test_uniform_sampling(self):
        buffer = replay_buffer.ReplayBuffer.initialize_empty(
            size=2, observation=jnp.array(1.), action=jnp.zeros((1,)))
        transitions = environment_lib.Transition(
            observation=jnp.array([1.]),
            action=jnp.array([[0]]),
            next_observation=jnp.array([2.]),
            reward=jnp.zeros([1]),
            done=jnp.zeros([1], dtype=bool))
        buffer = buffer.with_transitions(transitions)
        self.assertFalse(buffer.is_full)

        # Sampling from a partially-full buffer should use only the valid
        # entries.
        transition = buffer.sample_uniform(seed=test_util.test_seed(),
                                           batch_shape=())
        self.assertEqual(transition.observation.shape, ())

        transitions = buffer.sample_uniform(seed=test_util.test_seed(),
                                            batch_shape=[100])
        self.assertEqual(transitions.observation.shape, (100,))
        self.assertEqual(np.sum(transitions.observation == 1.), 100.)

        # Fill the buffer.
        more_transitions = environment_lib.Transition(
            observation=jnp.array([2., 3.]),
            action=jnp.array([[0], [0]]),
            next_observation=jnp.array([3., 4.]),
            reward=jnp.zeros([2]),
            done=jnp.zeros([2], dtype=bool))
        buffer = buffer.with_transitions(more_transitions)
        self.assertTrue(buffer.is_full)
        self.assertEqual(buffer.index, 1)

        transitions = buffer.sample_uniform(seed=test_util.test_seed(),
                                            batch_shape=[100])
        twos = np.sum(transitions.observation == 2.)
        threes = np.sum(transitions.observation == 3.)
        self.assertGreater(twos, 40)
        self.assertGreater(threes, 40)
        self.assertEqual(twos + threes, 100)
