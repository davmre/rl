from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np

from flax import struct

from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util


@struct.dataclass
class ReplayBuffer:

    transitions: environment_lib.Transition
    index: jnp.ndarray
    is_full: jnp.ndarray

    @staticmethod
    def initialize_empty(size, observation: jnp.ndarray, action: jnp.ndarray):

        def batch_zeros_like(s):
            s = jnp.asarray(s)
            return jnp.zeros((size,) + s.shape, dtype=s.dtype)

        dummy_observations = batch_zeros_like(observation)
        return ReplayBuffer(transitions=environment_lib.Transition(
            observation=dummy_observations,
            action=batch_zeros_like(action),
            next_observation=dummy_observations,
            reward=jnp.zeros([size]),
            done=jnp.zeros([size], dtype=bool)),
                            index=jnp.zeros([], dtype=jnp.int32),
                            is_full=jnp.zeros([], dtype=bool))

    @property
    def size(self):
        return self.transitions.action.shape[0]

    def reset(self):
        return ReplayBuffer(transitions=self.transitions,
                            index=jnp.zeros([], dtype=jnp.int32),
                            is_full=jnp.zeros([], dtype=bool))

    def with_transition(self, transition: environment_lib.Transition):
        return self.with_transitions(
            jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None, ...],
                                   transition))

    def with_transitions(self, transitions: environment_lib.Transition):
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

    def valid_transitions(self) -> environment_lib.Transition:
        """Returns the valid transitions in the buffer."""
        valid_idx = jnp.where(self.is_full, self.size, self.index)
        return jax.tree_util.tree_map(lambda x: x[:valid_idx], self.transitions)

    def sample_uniform(
        self, seed: type_util.KeyArray, batch_shape: tuple = ()
    ) -> environment_lib.Transition:
        """Samples `batch_size` transitions uniformly from the replay buffer."""
        indices = jax.random.randint(seed,
                                     shape=batch_shape,
                                     minval=0,
                                     maxval=jnp.where(self.is_full, self.size,
                                                      self.index))
        return jax.tree_util.tree_map(lambda x: x[indices], self.transitions)
