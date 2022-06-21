import dataclasses
from typing import Any, Callable, Dict, Optional, Union

import jax
from jax import numpy as jnp
import numpy as np

from brax import envs as brax_envs
from flax import struct

from daves_rl_lib.internal import type_util
from daves_rl_lib.internal import util


@dataclasses.dataclass
class ActionSpace:
    """Specifies the action space of an environment.

    Elements:
      shape: `tuple` of integers specifying the shape of an action.
      num_actions: optional `int` specifying the number of actions in a discrete
        action space. Note that discrete actions must (currently) have scalar
        shape; this is not explicitly checked but violations may lead to
        undefined behavior. If `None`, the action space is continuous.
    """
    shape: tuple = ()
    num_actions: Optional[int] = None

    @property
    def is_discrete(self):
        return self.num_actions is not None

    @staticmethod
    def from_gym_space(gym_space):
        if hasattr(gym_space, 'n'):
            if gym_space.start != 0:
                raise ValueError('Gym action space start must be 0.')
            return ActionSpace(num_actions=gym_space.n)
        raise NotImplementedError('Gym space type not supported:',
                                  type(gym_space))


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

    @staticmethod
    def from_brax_state(brax_state, num_steps, episode_return, seed):
        """Constructs a state representation from a Brax state."""
        return State(observation=brax_state.obs,
                     num_steps=num_steps,
                     seed=seed,
                     reward=brax_state.reward,
                     done=brax_state.reward,
                     unobserved=brax_state.qp,
                     episode_return=episode_return,
                     info=brax_state.info,
                     metrics=brax_state.metrics)

    def as_brax_state(self):
        """Represents a state as a Brax state."""
        return brax_envs.State(obs=self.observation,
                               reward=jnp.asarray(self.reward),
                               done=jnp.asarray(self.done),
                               qp=self.unobserved,
                               info=self.info,
                               metrics=self.metrics)  # type: ignore

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

    def __init__(self, action_space, discount_factor=1.0):
        self.action_space = action_space
        self.discount_factor = discount_factor

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


class BRAXEnvironment(Environment):
    """Wrapper to represent a BRAX environment."""

    def __init__(self, brax_env, discount_factor=1.0):
        self._brax_env = brax_env
        super().__init__(
            action_space=ActionSpace(shape=(brax_env.action_size,)),
            discount_factor=discount_factor)

    def _reset(self, seed: type_util.KeyArray) -> State:
        reset_seed, next_seed = jax.random.split(seed)
        return State.from_brax_state(self._brax_env.reset(reset_seed),
                                     num_steps=jnp.zeros([], dtype=jnp.int32),
                                     episode_return=jnp.zeros(
                                         [], dtype=jnp.float32),
                                     seed=next_seed)

    def _step(self, state: State, action: jnp.ndarray) -> State:
        new_state = self._brax_env.step(state.as_brax_state(), action)
        # Step and episode return are updated by the public caller.
        return State.from_brax_state(new_state,
                                     num_steps=state.num_steps,
                                     episode_return=state.episode_return,
                                     seed=state.seed)


class ExternalEnvironment(object):
    """Represents an environment with external state."""
    action_space: ActionSpace
    discount_factor: float = 1.0

    def __init__(self, action_space, discount_factor=1.0):
        self.action_space = action_space
        self.discount_factor = discount_factor

    @property
    def observation_size(self):
        raise NotImplementedError('observation_size is not implemented')

    def reset(self, seed=None):
        raise NotImplementedError('reset() not implemented.')

    def step(self, action):
        raise NotImplementedError('step() not implemented.')


class GymEnvironment(ExternalEnvironment):
    """Wraps an OpenAI Gym environment."""

    def __init__(self, gym_env, discount_factor=1.):
        self._gym_env = gym_env
        self._action_sequence = []
        self._initial_observation = None
        super().__init__(action_space=ActionSpace.from_gym_space(
            gym_env.action_space),
                         discount_factor=discount_factor)

    @property
    def observation_size(self):
        if self._initial_observation is None:
            self.reset()
        return self._initial_observation.shape[-1]  # type: ignore

    def step(self, action):
        action = np.asarray(action)
        self._action_sequence.append(action)
        observation, reward, done, info = self._gym_env.step(action=action)
        self._episode_return += self.discount_factor**self._num_steps * reward
        self._num_steps += 1
        return State(observation=observation,
                     reward=reward,
                     done=done,
                     info=info,
                     episode_return=self._episode_return,
                     num_steps=jnp.asarray(self._num_steps),
                     seed=0)

    def reset(self, seed=None, batch_size=None):
        if batch_size is not None:
            raise ValueError('Batching gym environments is not supported.')
        self._action_sequence = []
        self._episode_return = 0.
        self._num_steps = 0
        observation = self._gym_env.reset(seed=util.as_numpy_seed(seed))
        self._initial_observation = observation
        return State(observation=observation,
                     reward=jnp.asarray(0),
                     done=jnp.asarray(False),
                     info={},
                     episode_return=jnp.asarray(0.),
                     num_steps=jnp.asarray(0),
                     seed=0)