from typing import Any

import jax
import jax.numpy as jnp

from flax import struct


@struct.dataclass
class Transition:
    state: Any
    action: jnp.ndarray
    next_state: Any
    td_error: jnp.ndarray


@struct.dataclass
class ReplayBuffer:

    transitions: Transition
    index: jnp.ndarray
    is_full: jnp.ndarray

    @staticmethod
    def initialize_empty(size, dummy_state: Any, action_shape: tuple):

        def batch_zeros_like(s):
            s = jnp.asarray(s)
            return jnp.zeros((size,) + s.shape, dtype=s.dtype)

        dummy_states = jax.tree_util.tree_map(batch_zeros_like, dummy_state)
        return ReplayBuffer(transitions=Transition(
            state=dummy_states,
            action=jnp.zeros((size,) + action_shape),
            next_state=dummy_states,
            td_error=jnp.zeros([size])),
                            index=jnp.zeros([], dtype=jnp.int32),
                            is_full=jnp.zeros([], dtype=bool))

    @property
    def size(self):
        return self.transitions.action.shape[0]

    def with_transition(self, transition: Transition):
        return self.with_transitions(
            jax.tree_util.tree_map(lambda x: x[None, ...], transition))

    def with_transitions(self, transitions: Transition):
        """Returns a new buffer, replacing the oldest transitions."""
        num_transitions = transitions.action.shape[0]
        if num_transitions > self.size:
            raise ValueError(
                'Got {} transitions, but buffer only holds {}.'.format(
                    num_transitions, self.size))

        def add_transition_to_buffer(buffer, i):
            return jax.tree_util.tree_map(
                lambda x, t: x.at[(self.index + i) % self.size].set(t[i]),
                buffer, transitions), ()

        updated_transitions, _ = jax.lax.scan(add_transition_to_buffer,
                                              self.transitions,
                                              jnp.arange(num_transitions))
        updated_index = (self.index + num_transitions) % self.size
        updated_is_full = jnp.logical_or(
            self.is_full, self.index + num_transitions >= self.size)
        return ReplayBuffer(transitions=updated_transitions,
                            index=updated_index,
                            is_full=updated_is_full)

    def sample_uniform(self, seed, batch_shape=()) -> Transition:
        """Samples `batch_size` transitions uniformly from the replay buffer."""
        indices = jax.random.randint(seed,
                                     shape=batch_shape,
                                     minval=0,
                                     maxval=jnp.where(self.is_full, self.size,
                                                      self.index))
        return jax.tree_util.tree_map(lambda x: x[indices], self.transitions)
