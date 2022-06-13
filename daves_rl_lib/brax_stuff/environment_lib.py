import dataclasses
from typing import Any, Callable, Dict, Optional
import jax
from jax import numpy as jnp

import brax

import numpy as np

from flax import struct

from daves_rl_lib import util


@dataclasses.dataclass
class ActionSpace:
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
    observation: Any
    reward: jnp.ndarray
    done: jnp.ndarray
    seed: jnp.ndarray
    extra: Any = ()
    step: jnp.ndarray = 0
    episode_return: jnp.ndarray = 0.
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

    @staticmethod
    def from_brax_state(brax_state, step, episode_return, seed):
        return State(observation=brax_state.obs,
                     step=step,
                     seed=seed,
                     reward=brax_state.reward,
                     done=brax_state.reward,
                     extra=brax_state.qp,
                     episode_return=episode_return,
                     info=brax_state.info,
                     metrics=brax_state.metrics)

    def as_brax_state(self):
        return brax.State(obs=self.observation,
                          reward=self.reward,
                          done=self.done,
                          qp=self.extra,
                          info=self.info,
                          metrics=self.metrics)


class Environment(object):
    action_space: ActionSpace
    discount_factor: float = 1.0

    def __init__(self, action_space, discount_factor=1.0):
        self.action_space = action_space
        self.discount_factor = discount_factor


class PythonEnvironment(Environment):

    def reset(self, seed=None):
        raise NotImplementedError('reset() not implemented.')

    def step(self, action):
        raise NotImplementedError('step() not implemented.')


class GymEnvironment(PythonEnvironment):

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
        return self._initial_observation.shape[-1]

    def step(self, action):
        action = np.asarray(action)
        self._action_sequence.append(action)
        observation, reward, done, info = self._gym_env.step(action=action)
        self._episode_return += self.discount_factor**self._step * reward
        self._step += 1
        return State(observation=observation,
                     reward=reward,
                     done=done,
                     info=info,
                     episode_return=self._episode_return,
                     step=self._step,
                     seed=0)

    def reset(self, seed=None):
        self._action_sequence = []
        self._episode_return = 0.
        self._step = 0
        observation = self._gym_env.reset(seed=util.as_numpy_seed(seed))
        self._initial_observation = observation
        return State(observation=observation,
                     reward=0,
                     done=False,
                     info={},
                     episode_return=0,
                     step=0,
                     seed=0)


class JAXEnvironment(Environment):

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        reset_state = self.reset(seed=jax.random.PRNGKey(0))
        return reset_state.observation.shape[-1]

    def reset(self, seed) -> State:
        ts = self._reset(seed=seed)
        return dataclasses.replace(ts,
                                   step=jnp.zeros([], dtype=jnp.int32),
                                   episode_return=ts.reward)

    def reset_if_done(self, state: State) -> State:
        return util.tree_where(state.done, self.reset(seed=state.seed), state)

    def step(self, state: State, action: jnp.ndarray) -> State:
        new_state = self._step(state, action)
        new_state = dataclasses.replace(
            new_state,
            step=state.step + 1,
            episode_return=state.episode_return +
            self.discount_factor**(state.step) * new_state.reward)
        # Replace step with no-op if the episode was already done.
        return util.tree_where(
            state.done,
            dataclasses.replace(state, reward=jnp.zeros_like(state.reward)),
            new_state)

    def _reset(self, seed=None) -> State:
        raise NotImplementedError('reset() not implemented.')

    def _step(self, state, action) -> State:
        raise NotImplementedError('step() not implemented.')


class BRAXEnvironment(JAXEnvironment):

    def __init__(self, brax_env, discount_factor):
        self._brax_env = brax_env
        super().__init__(
            action_space=ActionSpace(shape=(brax_env.action_size,)),
            discount_factor=discount_factor)

    def _reset(self, seed) -> State:
        reset_seed, next_seed = jax.random.split(seed)
        return State.from_brax_state(self._brax_env.reset(reset_seed),
                                     step=jnp.zeros([], dtype=jnp.int32),
                                     episode_return=jnp.zeros(
                                         [], dtype=jnp.float32),
                                     seed=next_seed)

    def _step(self, state: State, action: jnp.ndarray) -> State:
        new_state = self._brax_env.step(state.as_brax_state(), action)
        # Step and episode return are updated by the public caller.
        return State.from_brax_state(new_state,
                                     step=state.step,
                                     episode_return=state.episode_return)


def initialize_batch(env, seed, batch_size=None):
    if batch_size:
        batch_seeds = jax.random.split(seed, batch_size)
        return jax.vmap(initialize_batch, in_axes=(None, 0))(env, batch_seeds)
    return env.reset(seed)