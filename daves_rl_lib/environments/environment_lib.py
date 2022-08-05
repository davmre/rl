from argparse import Action
import dataclasses
from typing import Any, Callable, Dict, Optional, Union

import jax
from jax import numpy as jnp
import numpy as np

from flax import struct

from daves_rl_lib.internal import type_util
from daves_rl_lib.internal import util


@struct.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    next_observation: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

    @property
    def batch_shape(self):
        return self.done.shape


@dataclasses.dataclass
class ActionSpace:
    """Specifies the action space of an environment.

    Elements:
      shape: `tuple` of integers specifying the shape of an action.
      minimum: optional array of shape `shape`, specifying the minimum value
        of a continuous-valued action. (currently, discrete actions are always
        integers in `{0, ..., num_actions - 1}`).
      maximum: optional array of shape `shape`, specifying the maximum value
        of a continuous-valued action. (currently, discrete actions are always
        integers in `{0, ..., num_actions - 1}`).
      num_actions: optional `int` specifying the number of actions in a discrete
        action space. Note that discrete actions must (currently) have scalar
        shape; this is not explicitly checked but violations may lead to
        undefined behavior. If `None`, the action space is continuous.
    """
    shape: tuple = ()
    minimum: Optional[jnp.ndarray] = None
    maximum: Optional[jnp.ndarray] = None
    num_actions: Optional[int] = None

    @property
    def dtype(self):
        return jnp.int32 if self.is_discrete else jnp.float32

    @property
    def is_discrete(self):
        return self.num_actions is not None

    def dummy_action(self, batch_size=None):
        return jnp.zeros(self.shape if batch_size is None else
                         (batch_size,) + self.shape,
                         dtype=self.dtype)


@struct.dataclass
class State:
    """Represents the current state of an environment.

    This may also represent a batch of states using a batch dimension in
    all of its element arrays.

    Elements:
        observation: the agent's observation following the most recent action.
        reward: scalar `float` reward received following the most recent action.
        done: `bool` indicating whether the episode has ended.
        seed: `jax.random.PRNGKey` random seed used to generate environmental
          randomness, i.e., the results of future `step` and `reset`
          invocations.
        unobserved: any additional information needed to specify state not
          present in the observations.
        num_steps: scalar `int` number of steps taken in the current episode.
        episode_return: scalar `float` sum of discounted rewards received in the
          current episode.
        metrics: `dict` containing additional metrics (for compatibility with
          Brax states.).
        info: `dict` of additional information about the current state (for
          compatibility with Brax states.).
    """
    observation: type_util.PyTree
    reward: jnp.ndarray
    done: jnp.ndarray
    seed: type_util.KeyArray
    num_steps: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0))
    episode_return: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array(0.))
    unobserved: type_util.PyTree = ()
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

    def minimize(self):
        """Remove debugging information from the state."""
        return dataclasses.replace(self, metrics={}, info={})

    @property
    def batch_shape(self):
        return self.seed.shape[:-1]


class Environment(object):
    """Represents an environment with fully-managed state.

    Environments must not depend on external state, so that they can be used
    in functional code and with JAX transformations (`jit`, `vmap`, etc.). Use
    the `ExternalEnvirnoment` class if you need to interact with external state.
    """
    action_space: ActionSpace
    discount_factor: float = 1.0

    def __init__(self,
                 action_space,
                 discount_factor=1.0,
                 max_episode_length: Optional[int] = None):
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.max_episode_length = max_episode_length

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        reset_state = self.reset(seed=jax.random.PRNGKey(0))
        return reset_state.observation.shape[-1]

    def reset(self, seed: type_util.KeyArray, batch_size=None) -> State:
        """Initializes an episode or batch of episodes.

        Args:
          seed: `jax.random.PRNGKey` random seed.
          batch_size: optional `int` number of independent states to initialize.
        Returns:
          state: `State` instance representing the initial state(s).
        """
        if batch_size:
            return jax.vmap(self.reset)(jax.random.split(seed, batch_size))
        ts = self._reset(seed=seed)
        return dataclasses.replace(ts,
                                   num_steps=jnp.zeros([], dtype=jnp.int32),
                                   episode_return=ts.reward)

    def reset_if_done(self, state: State) -> State:
        return util.tree_where(state.done, self.reset(seed=state.seed), state)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Advances an episode.

        This applies the action to the current state and returns the next state.
        If the current state is `done`, it is returned unchanged with no
        additional reward.

        Args:
          state: `State` instance representing the current episode state.
          action: `jnp.ndarray` representing the action to take.
        Returns:
          `new_state`: `State` instance representing the new episode state.
        """
        if state.batch_shape:
            raise ValueError(
                f'`step` was called on a batch of {state.batch_shape} states. '
                'This is not supported; please use an explicit `vmap`.')
        new_state = self._step(state, action)
        new_state = dataclasses.replace(
            new_state,
            num_steps=state.num_steps + 1,
            episode_return=state.episode_return +
            self.discount_factor**(state.num_steps) * new_state.reward)
        if self.max_episode_length is not None:
            new_state = dataclasses.replace(
                new_state,
                done=jnp.logical_or(
                    new_state.done,
                    new_state.num_steps >= self.max_episode_length))
        # Replace step with no-op if the episode was already done.
        return util.tree_where(
            state.done,
            dataclasses.replace(state, reward=jnp.zeros_like(state.reward)),
            new_state)

    def _reset(self, seed: type_util.KeyArray) -> State:
        """Initializes an episode (subclasses must implement)."""
        raise NotImplementedError('reset() not implemented.')

    def _step(self, state: State, action: jnp.ndarray) -> State:
        """Advances an episode (subclasses must implement)."""
        raise NotImplementedError('step() not implemented.')


class ExternalEnvironment(object):
    """Represents an environment with external state."""
    action_space: ActionSpace
    discount_factor: float = 1.0

    def __init__(self,
                 action_space: ActionSpace,
                 action_transform_fn: Optional[Callable] = None,
                 max_episode_length: Optional[int] = None,
                 discount_factor: float = 1.0):
        self.action_space = action_space
        self._action_sequence = []
        self._initial_observation = None
        self._num_steps = jnp.array(0)
        self._episode_return = jnp.array(0.)
        self.discount_factor = discount_factor
        self.action_transform_fn = (action_transform_fn
                                    if action_transform_fn else lambda x: x)
        self._max_episode_length = max_episode_length

    @property
    def observation_size(self):
        if self._initial_observation is None:
            self.reset()
        return self._initial_observation.shape[-1]  # type: ignore

    def reset(self, seed=None, batch_size=None):
        if batch_size is not None:
            raise ValueError('Batching external environments is not supported.')
        self._action_sequence = []
        state = self._reset(seed=seed)
        self._initial_observation = state.observation
        self._episode_return = jnp.array(0.)
        self._num_steps = jnp.array(0)
        return state

    def _reset(self, seed=None):
        raise NotImplementedError('_reset() not implemented.')

    def step(self, action):
        self._action_sequence.append(action)
        new_state = self._step(external_action=self.action_transform_fn(action))
        self._episode_return += (self.discount_factor**self._num_steps *
                                 new_state.reward)
        self._num_steps += 1
        done = new_state.done
        if self._max_episode_length is not None:
            done = jnp.logical_or(done,
                                  self._num_steps >= self._max_episode_length)
        return dataclasses.replace(new_state,
                                   done=done,
                                   episode_return=self._episode_return,
                                   num_steps=jnp.asarray(self._num_steps))

    def _step(self, external_action):
        raise NotImplementedError('_step() not implemented.')
