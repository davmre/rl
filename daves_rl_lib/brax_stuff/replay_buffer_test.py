from absl.testing import parameterized

import numpy as np
import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import test_util
from daves_rl_lib.brax_stuff import replay_buffer


def dummy_transitions(num_transitions, seed, action_is_vector=True):
    states = jax.random.randint(seed,
                                shape=[num_transitions],
                                minval=0,
                                maxval=1000)
    actions = jax.random.randint(
        seed,
        shape=[num_transitions, 1] if action_is_vector else [num_transitions],
        minval=-10,
        maxval=10)
    return replay_buffer.Transition(
        state=states,
        action=actions,
        next_state=states + (actions[..., 0] if action_is_vector else actions),
        td_error=jnp.zeros([num_transitions]))


class ReplayBufferTests(test_util.TestCase):

    @parameterized.named_parameters([('_scalar_action', False),
                                     ('_vector_action', True)])
    def test_initialize_and_fill_buffer(self, action_is_vector):
        buffer_size = 32
        buffer = replay_buffer.ReplayBuffer.initialize_empty(
            size=buffer_size,
            dummy_state=jnp.array(1),
            action_shape=(1,) if action_is_vector else ())
        self.assertEqual(buffer.size, buffer_size)
        self.assertEqual(buffer.transitions.state.shape, (buffer_size,))
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
            size=buffer_size, dummy_state=jnp.array(1), action_shape=())
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
        self.assertAllEqual(transitions.state[:5],
                            buffer.transitions.state[-5:])
        # Remaining transitions wrap around to the beginning.
        self.assertAllEqual(transitions.state[5:],
                            buffer.transitions.state[:buffer_size - 10])
        # The wraparound doesn't fully overwrite the previous entries, so they
        # should still be there.
        self.assertAllEqual(transitions.state[-5:],
                            buffer.transitions.state[-10:-5])

    def test_uniform_sampling(self):
        buffer = replay_buffer.ReplayBuffer.initialize_empty(
            size=2, dummy_state=jnp.array(1.), action_shape=(1,))
        transitions = replay_buffer.Transition(state=jnp.array([1.]),
                                               action=jnp.array([[0]]),
                                               next_state=jnp.array([2.]),
                                               td_error=jnp.zeros([1]))
        buffer = buffer.with_transitions(transitions)
        self.assertFalse(buffer.is_full)

        # Sampling from a partially-full buffer should use only the valid
        # entries.
        transition = buffer.sample_uniform(seed=test_util.test_seed(),
                                           batch_shape=())
        self.assertEqual(transition.state.shape, ())

        transitions = buffer.sample_uniform(seed=test_util.test_seed(),
                                            batch_shape=[100])
        self.assertEqual(transitions.state.shape, (100,))
        self.assertEqual(np.sum(transitions.state == 1.), 100.)

        # Fill the buffer.
        more_transitions = replay_buffer.Transition(
            state=jnp.array([2., 3.]),
            action=jnp.array([[0], [0]]),
            next_state=jnp.array([3., 4.]),
            td_error=jnp.zeros([2]))
        buffer = buffer.with_transitions(more_transitions)
        self.assertTrue(buffer.is_full)
        self.assertEqual(buffer.index, 1)

        transitions = buffer.sample_uniform(seed=test_util.test_seed(),
                                            batch_shape=[100])
        twos = np.sum(transitions.state == 2.)
        threes = np.sum(transitions.state == 3.)
        self.assertGreater(twos, 40)
        self.assertGreater(threes, 40)
        self.assertEqual(twos + threes, 100)

    def test_structured_state(self):
        dummy_state = {'a': jnp.array(1.), 'b': jnp.zeros([2, 3])}
        buffer = replay_buffer.ReplayBuffer.initialize_empty(
            size=7, dummy_state=dummy_state, action_shape=())
        self.assertShapeNested(buffer.transitions.state, {
            'a': [7],
            'b': [7, 2, 3]
        })
        self.assertShapeNested(buffer.transitions.next_state, {
            'a': [7],
            'b': [7, 2, 3]
        })
        self.assertEqual(buffer.transitions.action.shape, (7,))

        dummy_transition = replay_buffer.Transition(state=dummy_state,
                                                    next_state=dummy_state,
                                                    action=jnp.zeros([]),
                                                    td_error=jnp.zeros([]))
        buffer = buffer.with_transition(dummy_transition)
        sampled_transition = buffer.sample_uniform(seed=test_util.test_seed(),
                                                   batch_shape=())
        self.assertAllEqualNested(dummy_transition, sampled_transition)