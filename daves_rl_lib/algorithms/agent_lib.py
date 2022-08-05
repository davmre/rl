import dataclasses
from os import environ
from typing import Any, Optional

import jax
from jax import numpy as jnp

from flax import struct
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util


class Agent(object):

    def init_weights(self,
                     seed: type_util.KeyArray,
                     dummy_observation: Optional[jnp.ndarray] = None,
                     dummy_action: Optional[jnp.ndarray] = None,
                     **kwargs):
        return self._init_weights(seed,
                                  dummy_observation=dummy_observation,
                                  dummy_action=dummy_action,
                                  **kwargs)

    def _init_weights(self,
                      seed: type_util.KeyArray,
                      dummy_observation: Optional[jnp.ndarray],
                      dummy_action: Optional[jnp.ndarray],
                      batch_size: Optional[int] = None) -> Any:
        raise NotImplementedError()

    def action_dist(self, weights,
                    observation: jnp.ndarray) -> tfp.distributions.Distribution:
        return self._action_dist(weights, observation)

    def _action_dist(self, weights, observation: jnp.ndarray):
        raise NotImplementedError()

    def update(self, weights: Any, transition: environment_lib.Transition,
               **kwargs) -> Any:
        return self._update(weights, transition, **kwargs)

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
                     dummy_observation: jnp.ndarray, dummy_action: jnp.ndarray,
                     **kwargs) -> EpisodicMemoryWeights:
        return EpisodicMemoryWeights(
            episode_buffer=replay_buffer.ReplayBuffer.initialize_empty(
                size=self._max_num_steps,
                observation=dummy_observation,
                action=dummy_action),
            agent_weights=self._init_weights(
                seed,
                dummy_observation=dummy_observation,
                dummy_action=dummy_action,
                **kwargs))

    def action_dist(self, weights: EpisodicMemoryWeights,
                    observation: jnp.ndarray) -> tfp.distributions.Distribution:
        return self._action_dist(weights.agent_weights, observation)

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


def zero_out_suffix_of_elements(x, num_valid):
    num_total = x.shape[0]
    valid_mask = jnp.arange(num_total) < num_valid
    return jnp.where(jnp.reshape(valid_mask, (-1,) + (1,) * (len(x.shape) - 1)),
                     x, jnp.zeros_like(x))


@struct.dataclass
class PeriodicUpdateAgentWeights:
    steps_buffer: replay_buffer.ReplayBuffer
    num_updates: jnp.ndarray
    agent_weights: Any


class PeriodicUpdateAgent(Agent):
    """Agent that updates its weights every `steps_per_update` steps."""

    def __init__(self, steps_per_update: int):
        self._steps_per_update = steps_per_update
        super().__init__()

    def init_weights(self,
                     seed: type_util.KeyArray,
                     dummy_observation: jnp.ndarray,
                     dummy_action: jnp.ndarray,
                     batch_size: Optional[int] = None,
                     **kwargs) -> PeriodicUpdateAgentWeights:
        init_steps_buffer = (lambda _: replay_buffer.ReplayBuffer.
                             initialize_empty(size=self._steps_per_update,
                                              observation=dummy_observation,
                                              action=dummy_action))
        return PeriodicUpdateAgentWeights(
            steps_buffer=(jax.vmap(init_steps_buffer)(jnp.arange(batch_size))
                          if batch_size else init_steps_buffer(None)),
            num_updates=jnp.zeros([], dtype=jnp.int32),
            agent_weights=self._init_weights(seed,
                                             batch_size=batch_size,
                                             
                                             dummy_observation=dummy_observation,
                                             dummy_action=dummy_action,
                                             **kwargs))

    def action_dist(self, weights: PeriodicUpdateAgentWeights,
                    observation: jnp.ndarray) -> tfp.distributions.Distribution:
        return self._action_dist(weights.agent_weights, observation)

    def update(
            self, weights: PeriodicUpdateAgentWeights,
            transition: environment_lib.Transition
    ) -> PeriodicUpdateAgentWeights:
        reset_buffer_fn = lambda b: b.reset()
        if transition.done.shape:  # parallel batch of transitions
            steps_buffer = jax.vmap(lambda b, t: b.with_transition(t))(
                weights.steps_buffer, transition)
            reset_buffer_fn = jax.vmap(reset_buffer_fn)
        else:
            steps_buffer = weights.steps_buffer.with_transition(transition)

        return jax.lax.cond(
            # Formally check if any worker's buffer is full, though since all
            # buffers are the same size they must fill (and be reset)
            # at the same time.
            jnp.any(weights.steps_buffer.is_full),
            # Update the agent's weights and reset the buffer.
            lambda: dataclasses.replace(
                weights,
                agent_weights=self._update(weights.agent_weights, weights.
                                           steps_buffer.transitions),
                steps_buffer=reset_buffer_fn(steps_buffer),
                num_updates=weights.num_updates + 1),
            lambda: dataclasses.replace(weights, steps_buffer=steps_buffer))

    def _update(self, weights: Any,
                transitions: environment_lib.Transition) -> Any:
        raise NotImplementedError()


@struct.dataclass
class ReplayBufferAgentWeights:
    replay_buffer: replay_buffer.ReplayBuffer
    agent_weights: Any
    seed: type_util.KeyArray


class ReplayBufferAgent(Agent):
    """Agent that updates its weights every `steps_per_update` steps."""

    def __init__(self, replay_buffer_size: int, gradient_batch_size: int):
        self._replay_buffer_size = replay_buffer_size
        self._gradient_batch_size = gradient_batch_size
        super().__init__()

    def init_weights(self, seed: type_util.KeyArray,
                     dummy_observation: jnp.ndarray, dummy_action: jnp.ndarray,
                     **kwargs) -> ReplayBufferAgentWeights:
        seed, weights_seed = jax.random.split(seed, 2)
        return ReplayBufferAgentWeights(
            replay_buffer=replay_buffer.ReplayBuffer.initialize_empty(
                size=self._replay_buffer_size,
                observation=dummy_observation,
                action=dummy_action),
            agent_weights=self._init_weights(weights_seed, 
                                             dummy_observation=dummy_observation,
                                             dummy_action=dummy_action,
                                             **kwargs),
            seed=seed)

    def action_dist(self, weights: ReplayBufferAgentWeights,
                    observation: jnp.ndarray) -> tfp.distributions.Distribution:
        return self._action_dist(weights.agent_weights, observation)

    def update(
            self, weights: ReplayBufferAgentWeights,
            transition: environment_lib.Transition) -> ReplayBufferAgentWeights:
        batch_shape = transition.done.shape
        if batch_shape:
            replay_buffer = weights.replay_buffer.with_transitions(transition)
        else:
            replay_buffer = weights.replay_buffer.with_transition(transition)
        seed, replay_seed, next_seed = jax.random.split(weights.seed, 3)
        transitions = replay_buffer.sample_uniform(
            batch_shape=(self._gradient_batch_size,), seed=replay_seed)

        return dataclasses.replace(
            weights,
            agent_weights=self._update(
                weights.agent_weights,
                transitions=weights.replay_buffer.transitions,
                seed=seed),
            replay_buffer=replay_buffer,
            seed=next_seed)

    def _update(self, weights: Any, transitions: environment_lib.Transition,
                seed: type_util.KeyArray) -> Any:
        raise NotImplementedError()


def episode_reward_to_go(rewards: jnp.ndarray,
                         done: jnp.ndarray,
                         discount_factor: float,
                         final_state_value=None):
    if final_state_value is not None:
        # Treat the (discounted) value estimate for the final state as a reward
        # propagated to previous steps in the same episode.
        rewards = jnp.concatenate([rewards,
                                   jnp.array([final_state_value])
                                  ])  # type: ignore
        done = jnp.concatenate([done, jnp.array([False])])  # type: ignore

    size = rewards.shape[0]
    indices = jnp.arange(size)
    num_future_dones = jnp.cumsum(done[::-1])[::-1]

    relevance_square = jnp.logical_and(
        indices >= indices[..., None],
        num_future_dones == num_future_dones[..., None])

    episode_discount_square = jnp.cumprod(jnp.where(
        indices > indices[..., None], discount_factor, 1.),
                                          axis=-1) * relevance_square
    # Equivalent to alternative:
    # episode_discount_square = discount_factor**(
    #    indices - indices[..., None]) * relevance_square
    # but appears to be slightly faster for size=1000.
    reward_to_go = jnp.sum(episode_discount_square * rewards[None, ...],
                           axis=-1)

    if final_state_value is not None:
        # Remove the final state value.
        reward_to_go = reward_to_go[:-1]
    return reward_to_go


def update_moving_average(moving_average, value, decay):
    return jax.tree_util.tree_map(lambda x, y: decay * x + (1 - decay) * y,
                                  moving_average, value)


def tree_norm(grad):
    sqnorms = [jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)]
    return jnp.sqrt(sum(sqnorms))