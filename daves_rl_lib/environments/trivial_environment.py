import dataclasses
from typing import Any
import jax
from jax import numpy as jnp

import numpy as np

from flax import struct

from daves_rl_lib.environments import environment_lib


class OneStepEnvironment(environment_lib.Environment):

    def __init__(self, discount_factor=1.):
        super().__init__(action_space=environment_lib.ActionSpace(
            shape=(), num_actions=1),
                         discount_factor=discount_factor)

    def _reset(self, seed):
        return environment_lib.State(done=jnp.zeros([], dtype=bool),
                                     reward=jnp.zeros([]),
                                     observation=jnp.zeros([1]),
                                     seed=seed)

    def step(self, state, action):
        del action  # Unused.
        return dataclasses.replace(state,
                                   observation=jnp.ones_like(state.observation),
                                   done=jnp.ones([], dtype=bool),
                                   reward=jnp.where(state.done, 0., 1.))


class DiscreteTargetEnvironment(environment_lib.Environment):

    def __init__(self,
                 size=1,
                 dim=1,
                 one_hot_features=False,
                 discount_factor=1.):
        self._size = size
        self._dim = dim
        self._one_hot_features = one_hot_features
        super().__init__(
            action_space=environment_lib.ActionSpace(num_actions=dim * 2),
            discount_factor=discount_factor)

    @property
    def width(self):
        return self._size * 2 + 1

    def _to_state_idx(self, obs):
        return jnp.sum(
            (obs + self._size) * self.width**jnp.arange(self._dim - 1, -1, -1))

    def _from_state_idx(self, idx):
        coords = []
        for _ in range(self._dim):
            coords = [idx % self.width - self._size] + coords
            idx //= self.width
        return jnp.asarray(coords)

    def _to_features(self, pos):
        if self._one_hot_features:
            return jax.nn.one_hot(self._to_state_idx(pos),
                                  num_classes=self.width**self._dim)
        return pos

    def _from_features(self, features):
        if self._one_hot_features:
            return self._from_state_idx(jnp.argmax(features, axis=-1))
        return features

    def _reset(self, seed):
        return environment_lib.State(done=jnp.array(False),
                                     reward=jnp.zeros([]),
                                     observation=self._to_features(
                                         jnp.zeros([self._dim],
                                                   dtype=jnp.int32)),
                                     seed=seed)

    def _step(self, state: environment_lib.State, action: jnp.ndarray):
        # action is an int between 0 and dim*2.
        state_pos = self._from_features(state.observation)
        action_dim = action % self._dim
        action_dir = (action // self._dim) * 2 - 1
        delta = action_dir * jax.nn.one_hot(
            action_dim, num_classes=self._dim, dtype=state_pos.dtype)
        new_state_pos = state_pos + delta
        new_state_pos = jnp.minimum(
            jnp.maximum(new_state_pos,
                        -self._size * jnp.ones_like(new_state_pos)),
            self._size * jnp.ones_like(new_state_pos))
        new_state_done = jnp.all(new_state_pos == self._size *
                                 jax.nn.one_hot(0, num_classes=self._dim))
        return dataclasses.replace(state,
                                   observation=self._to_features(new_state_pos),
                                   done=new_state_done,
                                   reward=jnp.where(new_state_done, 1., 0.))


class ContinuousEnvironmentStateless(environment_lib.Environment):

    def __init__(self, dim=1, size=100., discount_factor=1.):
        self._dim = dim
        self._size = size
        super().__init__(action_space=environment_lib.ActionSpace(
            shape=(dim,), num_actions=None),
                         discount_factor=discount_factor)

    def _reset(self, seed):
        seed, next_seed = jax.random.split(seed)
        return environment_lib.State(done=jnp.array(False),
                                     reward=jnp.zeros([]),
                                     observation=self._size *
                                     jax.random.normal(seed, shape=[self._dim]),
                                     seed=next_seed)

    def _step(self, state, action):
        # action is a vector of shape [dim]
        loss = jnp.linalg.norm(action - state.observation)**2
        return dataclasses.replace(state,
                                   observation=state.observation,
                                   done=jnp.ones([], dtype=bool),
                                   reward=-loss)


class ContinuousEnvironmentInvertMatrix(environment_lib.Environment):

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
        super().__init__(action_space=environment_lib.ActionSpace(
            shape=(dim,), num_actions=None))

    def _reset(self, seed):
        seed, next_seed = jax.random.split(seed)
        return environment_lib.State(
            done=jnp.array(False),
            reward=jnp.zeros([]),
            seed=next_seed,
            observation=(self._size *
                         jax.random.normal(seed, shape=[self._dim])))

    def _step(self, state, action):
        # action is a vector of shape [dim]
        new_state_pos = state.observation + jnp.dot(self._matrix, action)
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
            reward += jnp.linalg.norm(state.observation) - distance_to_goal
        return dataclasses.replace(state,
                                   observation=new_state_pos,
                                   reward=reward,
                                   done=new_state_done)
