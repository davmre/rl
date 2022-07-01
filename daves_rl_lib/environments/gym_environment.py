from argparse import Action
import dataclasses
from os import environ
from typing import Any, Callable, Dict, Optional, Union

import jax
from jax import numpy as jnp
import numpy as np

from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import util


class GymEnvironment(environment_lib.ExternalEnvironment):
    """Wraps an OpenAI Gym environment."""

    def __init__(self,
                 gym_env,
                 discount_factor=1.,
                 max_episode_length: Optional[int] = None,
                 action_space: Optional[environment_lib.ActionSpace] = None,
                 action_transform_fn: Optional[Callable] = None):
        self._gym_env = gym_env
        if not action_space:
            if action_transform_fn:
                raise ValueError(
                    'Cannot infer action space when an action transform is '
                    'specified. Please specify the action space explicitly.')
            action_space = action_space_from_gym_space(gym_env.action_space)
        super().__init__(action_space=action_space,
                         action_transform_fn=action_transform_fn,
                         max_episode_length=max_episode_length,
                         discount_factor=discount_factor)

    def _step(self, external_action):
        external_action = np.asarray(
            external_action)  # Gym can't handle JAX arrays.
        observation, reward, done, info = self._gym_env.step(
            action=external_action)
        return environment_lib.State(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            episode_return=self._episode_return,  # Updated in public wrapper.
            num_steps=self._num_steps,  # Updated in public wrapper.
            seed=0)

    def _reset(self, seed=None):
        observation = self._gym_env.reset(seed=util.as_numpy_seed(seed))
        return environment_lib.State(observation=observation,
                                     reward=jnp.asarray(0),
                                     done=jnp.asarray(False),
                                     info={},
                                     episode_return=jnp.asarray(0.),
                                     num_steps=jnp.asarray(0),
                                     seed=0)


def action_space_from_gym_space(gym_space) -> environment_lib.ActionSpace:
    if hasattr(gym_space, 'n'):
        if gym_space.start != 0:
            raise ValueError('Gym action space start must be 0.')
        return environment_lib.ActionSpace(num_actions=gym_space.n)
    raise NotImplementedError('Gym space type not supported:', type(gym_space))
