from typing import Any, Callable, Dict, Optional, Union

import jax
from jax import numpy as jnp
import numpy as np

from brax import envs as brax_envs

from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util


class BRAXEnvironment(environment_lib.Environment):
    """Wrapper to represent a BRAX environment."""

    def __init__(self,
                 brax_env,
                 discount_factor=1.0,
                 action_space: Optional[environment_lib.ActionSpace] = None,
                 action_transform_fn: Optional[Callable] = None):
        self._brax_env = brax_env

        if not action_space:
            if action_transform_fn:
                raise ValueError(
                    'Cannot infer action space when an action transform is '
                    'specified. Please specify the action space explicitly.')
            action_space = environment_lib.ActionSpace(
                shape=(brax_env.action_size,))
        self.action_transform_fn = (action_transform_fn
                                    if action_transform_fn else lambda x: x)
        super().__init__(action_space=action_space,
                         discount_factor=discount_factor)

    def _reset(self, seed: type_util.KeyArray) -> environment_lib.State:
        reset_seed, next_seed = jax.random.split(seed)
        return from_brax_state(self._brax_env.reset(reset_seed),
                               num_steps=jnp.zeros([], dtype=jnp.int32),
                               episode_return=jnp.zeros([], dtype=jnp.float32),
                               seed=next_seed)

    def _step(self, state: environment_lib.State,
              action: jnp.ndarray) -> environment_lib.State:
        new_state = self._brax_env.step(as_brax_state(state),
                                        self.action_transform_fn(action))
        # Step and episode return are updated by the public caller.
        return from_brax_state(new_state,
                               num_steps=state.num_steps,
                               episode_return=state.episode_return,
                               seed=state.seed)


def from_brax_state(brax_state: brax_envs.State, num_steps, episode_return,
                    seed):
    """Constructs a state representation from a Brax state."""
    return environment_lib.State(
        observation=brax_state.obs,
        num_steps=num_steps,
        seed=seed,
        reward=jnp.asarray(brax_state.reward),
        done=jnp.asarray(brax_state.done),
        unobserved=brax_state.qp,
        episode_return=episode_return,
        info=brax_state.info,
        metrics=brax_state.metrics  # type: ignore
    )


def as_brax_state(state: environment_lib.State):
    """Represents a state as a Brax state."""
    return brax_envs.State(obs=state.observation,
                           reward=jnp.asarray(state.reward),
                           done=jnp.asarray(state.done),
                           qp=state.unobserved,
                           info=state.info,
                           metrics=state.metrics)  # type: ignore
