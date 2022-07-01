from typing import Any, Callable, Dict, Optional, Union

from dm_control.rl.control import dm_env as dm_env_lib

import jax
from jax import numpy as jnp
import numpy as np

from daves_rl_lib.environments import environment_lib


class DMControlEnvironment(environment_lib.ExternalEnvironment):
    """Wraps a Deepmind Control environment."""

    def __init__(self,
                 dm_env: dm_env_lib.Environment,
                 discount_factor=1.,
                 max_episode_length: Optional[int] = None,
                 action_space: Optional[environment_lib.ActionSpace] = None,
                 action_transform_fn: Optional[Callable] = None):
        self._dm_env = dm_env
        if not action_space:
            if action_transform_fn:
                raise ValueError(
                    'Cannot infer action space when an action transform is '
                    'specified. Please specify the action space explicitly.')
            action_space = action_space_from_dm_control_spec(
                dm_env.action_spec())
        super().__init__(action_space=action_space,
                         action_transform_fn=action_transform_fn,
                         max_episode_length=max_episode_length,
                         discount_factor=discount_factor)

    def _flatten_observation(self, observation: Any):
        # Concatenate all observed quantities in a single feature vector.
        return jnp.concatenate([
            jnp.reshape(x, [-1]) for x in jax.tree_util.tree_leaves(observation)
        ],
                               axis=0)

    def _state_from_timestep(self, timestep: dm_env_lib.TimeStep):
        return environment_lib.State(observation=self._flatten_observation(
            timestep.observation),
                                     reward=timestep.reward,
                                     done=jnp.zeros([], dtype=bool),
                                     num_steps=jnp.asarray(self._num_steps),
                                     episode_return=self._episode_return,
                                     seed=0)

    def _step(self, external_action):
        timestep = self._dm_env.step(external_action)
        return self._state_from_timestep(timestep)

    def _reset(self, seed=None):
        return self._state_from_timestep(self._dm_env.reset())


def action_space_from_dm_control_spec(
        action_spec) -> environment_lib.ActionSpace:
    return environment_lib.ActionSpace(
        shape=action_spec.shape,
        num_actions=None,  # dm_control tasks are continuous.
        minimum=action_spec.minimum,
        maximum=action_spec.maximum)