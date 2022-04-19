from typing import Any
import jax
from jax import numpy as jnp

import numpy as np

from flax import struct


@struct.dataclass
class TargetState:
    done: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray


class OneStepEnvironment(object):

    def __init__(self):
        self.action_size = 1

    def reset(self, seed=None):
        return TargetState(done=jnp.zeros([], dtype=np.bool),
                           reward=jnp.zeros([]),
                           obs=jnp.zeros([1]))

    def step(self, state, action):
        del action  # Unused.
        return TargetState(obs=jnp.ones_like(state.obs),
                           done=jnp.ones([], dtype=np.bool),
                           reward=jnp.where(state.done, 0., 1.))


class DiscreteTargetEnvironment(object):

    def __init__(self, size=1, dim=1):
        self._size = size
        self._dim = dim
        self.action_size = dim * 2

    def reset(self, seed=None):
        return TargetState(done=False,
                           reward=jnp.zeros([]),
                           obs=jnp.zeros([self._dim], dtype=jnp.int32))

    def step(self, state, action):
        # action is an int between 0 and dim*2.
        action_dim = action % self._dim
        action_dir = (action // self._dim) * 2 - 1
        delta = action_dir * jax.nn.one_hot(
            action_dim, num_classes=self._dim, dtype=state.obs.dtype)
        new_state_pos = state.obs + delta
        new_state_pos = jnp.minimum(
            jnp.maximum(new_state_pos,
                        -self._size * jnp.ones_like(new_state_pos)),
            self._size * jnp.ones_like(new_state_pos))
        new_state_done = jnp.all(new_state_pos == self._size *
                                 jax.nn.one_hot(0, num_classes=self._dim))
        return TargetState(obs=jnp.where(state.done, state.obs, new_state_pos),
                           done=jnp.where(state.done, state.done,
                                          new_state_done),
                           reward=jnp.where(state.done, 0.,
                                            jnp.where(new_state_done, 1., 0.)))


class ContinuousEnvironmentStateless(object):

    def __init__(self, dim=1, size=100.):
        self._dim = dim
        self._size = size

    def reset(self, seed):
        return TargetState(done=False,
                           reward=jnp.zeros([]),
                           obs=self._size *
                           jax.random.normal(seed, shape=[self._dim]))

    def step(self, state, action):
        # action is a vector of shape [dim]
        loss = jnp.linalg.norm(action - state.obs[:-1])**2
        return TargetState(obs=state.obs,
                           done=jnp.ones([], dtype=np.bool),
                           reward=-loss)


class ContinuousEnvironmentInvertMatrix(object):

    def __init__(self,
                 size=2,
                 dim=1,
                 goal_tolerance=0.5,
                 cost_of_living=0.1,
                 shape_reward=True):
        self._size = size
        self._dim = dim
        self._goal_tolerance = goal_tolerance
        self._shape_reward = shape_reward
        self._cost_of_living = cost_of_living
        self._matrix = jax.random.normal(jax.random.PRNGKey(0),
                                         shape=[self._dim, self._dim])

    def reset(self, seed):
        return TargetState(done=False,
                           reward=jnp.zeros([]),
                           obs=self._size *
                           jax.random.normal(seed, shape=[self._dim]))

    def step(self, state, action):
        # action is a vector of shape [dim]
        new_state_pos = state.obs + jnp.dot(self._matrix, action)
        new_state_pos = jnp.minimum(
            jnp.maximum(new_state_pos,
                        -self._size * 10 * jnp.ones_like(new_state_pos)),
            self._size * 10 * jnp.ones_like(new_state_pos))
        distance_to_goal = jnp.linalg.norm(new_state_pos)
        new_state_done = distance_to_goal < self._goal_tolerance
        reward = jnp.where(new_state_done,
                           1. - distance_to_goal / self._goal_tolerance,
                           -self._cost_of_living)
        if self._shape_reward:
            reward += jnp.linalg.norm(state.obs) - distance_to_goal
        return TargetState(obs=jnp.where(state.done, state.obs, new_state_pos),
                           done=jnp.where(state.done, state.done,
                                          new_state_done),
                           reward=jnp.where(state.done, 0., reward))
