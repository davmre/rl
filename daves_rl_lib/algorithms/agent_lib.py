from typing import Any

import jax
from jax import numpy as jnp

from flax import struct
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util


class Agent(object):

    def init_weights(self, seed: type_util.KeyArray,
                     dummy_observation: jnp.ndarray, dummy_action: jnp.ndarray):
        return self._init_weights(seed,
                                  dummy_observation=dummy_observation,
                                  dummy_action=dummy_action)

    def _init_weights(self, seed: type_util.KeyArray,
                      dummy_observation: jnp.ndarray,
                      dummy_action: jnp.ndarray):
        raise NotImplementedError()

    def action_dist(self, observation: jnp.ndarray,
                    weights) -> tfp.distributions.Distribution:
        return self._action_dist(observation, weights)

    def _action_dist(self, observation: jnp.ndarray, weights):
        raise NotImplementedError()

    def update(self, weights: Any,
               transition: environment_lib.Transition) -> Any:
        return self._update(weights, transition)

    def _update(self, weights, transition):
        raise NotImplementedError()


@struct.dataclass
class EpisodicMemoryWeights:
    episode_buffer: replay_buffer.ReplayBuffer
    agent_weights: Any


class EpisodicAgent(Agent):

    def __init__(self, max_num_steps: int):
        self._max_num_steps = max_num_steps
        super().__init__()

    def init_weights(self, seed: type_util.KeyArray,
                     dummy_observation: jnp.ndarray,
                     dummy_action: jnp.ndarray) -> EpisodicMemoryWeights:
        """
        If the agent will be run in a batch environment, the dummy observation and
        action should have the corresponding batch shape.
        """
        return EpisodicMemoryWeights(
            episode_buffer=replay_buffer.ReplayBuffer.initialize_empty(
                size=self._max_num_steps,
                observation=dummy_observation,
                action=dummy_action),
            agent_weights=self._init_weights(
                seed,
                dummy_observation=dummy_observation,
                dummy_action=dummy_action))

    def action_dist(
            self, observation: jnp.ndarray,
            weights: EpisodicMemoryWeights) -> tfp.distributions.Distribution:
        return self._action_dist(observation, weights.agent_weights)

    def update(self, weights: EpisodicMemoryWeights,
               transition: environment_lib.Transition) -> EpisodicMemoryWeights:
        batch_shape = transition.done.shape
        if batch_shape:
            raise NotImplementedError(
                "Batch transitions are not currently supported for episodic agents."
            )
        episode_buffer = weights.episode_buffer.with_transition(transition)

        def if_done():
            agent_weights = self._update_episode(
                weights.agent_weights,
                episode_buffer.transitions,
                num_valid_transitions=episode_buffer.index)
            return EpisodicMemoryWeights(agent_weights=agent_weights,
                                         episode_buffer=episode_buffer.reset())

        def if_not_done():
            return EpisodicMemoryWeights(agent_weights=weights.agent_weights,
                                         episode_buffer=episode_buffer)

        return jax.lax.cond(transition.done, if_done, if_not_done)

    def _update_episode(self, weights, transitions, num_valid_transitions):
        raise NotImplementedError()


def zero_invalid(x, num_valid):
    num_total = x.shape[0]
    valid_mask = jnp.arange(num_total) < num_valid
    return jnp.where(jnp.reshape(valid_mask, (-1,) + (1,) * (len(x.shape) - 1)),
                     x, jnp.zeros_like(x))
