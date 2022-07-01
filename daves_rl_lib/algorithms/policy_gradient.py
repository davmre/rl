import dataclasses
from email import policy
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from flax import struct
import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import networks
from daves_rl_lib.algorithms import agent_lib
from daves_rl_lib.algorithms import exploration_lib
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util
from daves_rl_lib.internal import util


@struct.dataclass
class PolicyGradientAgentWeights:
    policy_weights: type_util.PyTree
    policy_optimizer_state: type_util.PyTree


class EpisodicPolicyGradientAgent(agent_lib.EpisodicAgent):

    def __init__(
        self,
        policy_net: networks.FeedForwardModel,
        policy_optimizer: optax.GradientTransformation,
        max_num_steps: int,
        discount_factor: float = 1.,
        reward_to_go: bool = True,
        standardize_advantages: bool = True,
    ):
        self._policy_net = policy_net
        self._policy_optimizer = policy_optimizer
        self._discount_factor = discount_factor
        self._reward_to_go = reward_to_go
        self._standardize_advantages = standardize_advantages
        super().__init__(max_num_steps=max_num_steps)

    @property
    def policy_net(self):
        return self._policy_net

    def _init_weights(self, seed: type_util.KeyArray, **kwargs):
        policy_weights = self.policy_net.init(seed)
        return PolicyGradientAgentWeights(
            policy_weights=policy_weights,
            policy_optimizer_state=self._policy_optimizer.init(policy_weights))

    def _action_dist(
            self, weights: PolicyGradientAgentWeights,
            observation: jnp.ndarray) -> tfp.distributions.Distribution:
        return self.policy_net.apply(weights.policy_weights, observation)

    def _update_episode(
            self, weights: PolicyGradientAgentWeights,
            transitions: environment_lib.Transition,
            num_valid_transitions: jnp.ndarray) -> PolicyGradientAgentWeights:
        value_estimates = agent_lib.episode_reward_to_go(
            agent_lib.zero_out_suffix_of_elements(transitions.reward,
                                                  num_valid_transitions),
            done=jnp.zeros(transitions.reward.shape, dtype=bool),
            discount_factor=self._discount_factor)

        def surrogate_loss(policy_weights):
            action_dists = self.policy_net.apply(policy_weights,
                                                 transitions.observation)
            lps = agent_lib.zero_out_suffix_of_elements(
                action_dists.log_prob(transitions.action),
                num_valid_transitions)
            return -jnp.sum(lps * value_estimates)

        policy_gradient = jax.grad(surrogate_loss)(weights.policy_weights)
        policy_updates, policy_optimizer_state = self._policy_optimizer.update(
            policy_gradient, weights.policy_optimizer_state)
        return PolicyGradientAgentWeights(
            policy_weights=optax.apply_updates(weights.policy_weights,
                                               policy_updates),
            policy_optimizer_state=policy_optimizer_state)


class PolicyGradientAgent(agent_lib.PeriodicUpdateAgent):

    def __init__(
        self,
        policy_net: networks.FeedForwardModel,
        policy_optimizer: optax.GradientTransformation,
        steps_per_update: int,
        discount_factor: float = 1.,
        reward_to_go: bool = True,
        standardize_advantages: bool = True,
    ):
        self._policy_net = policy_net
        self._policy_optimizer = policy_optimizer
        self._discount_factor = discount_factor
        self._reward_to_go = reward_to_go
        self._standardize_advantages = standardize_advantages
        super().__init__(steps_per_update=steps_per_update)

    @property
    def policy_net(self):
        return self._policy_net

    def _init_weights(self, seed: type_util.KeyArray, **kwargs):
        policy_weights = self.policy_net.init(seed)
        return PolicyGradientAgentWeights(
            policy_weights=policy_weights,
            policy_optimizer_state=self._policy_optimizer.init(policy_weights))

    def _action_dist(
            self, weights: PolicyGradientAgentWeights,
            observation: jnp.ndarray) -> tfp.distributions.Distribution:
        return self.policy_net.apply(weights.policy_weights, observation)

    def _estimate_q_values(
            self, weights: PolicyGradientAgentWeights,
            transitions: environment_lib.Transition) -> jnp.ndarray:
        if self._reward_to_go:
            estimated_returns = agent_lib.episode_reward_to_go(
                rewards=transitions.reward,
                done=transitions.done,
                discount_factor=self._discount_factor)
        else:
            estimated_returns = episode_reward(
                rewards=transitions.reward,
                done=transitions.done,
                discount_factor=self._discount_factor)
        return estimated_returns

    def _estimate_advantages(
            self, weights: PolicyGradientAgentWeights,
            transitions: environment_lib.Transition) -> jnp.ndarray:
        advantages = self._estimate_q_values(weights=weights,
                                             transitions=transitions)

        if self._standardize_advantages:
            advantages = (advantages -
                          jnp.mean(advantages)) / (jnp.std(advantages) + 1e-6)
        return advantages

    def _update(
            self, weights: PolicyGradientAgentWeights,
            transitions: environment_lib.Transition
    ) -> PolicyGradientAgentWeights:
        advantage_estimates = self._estimate_advantages(weights=weights,
                                                        transitions=transitions)

        def surrogate_loss(policy_weights):
            action_dists = self.policy_net.apply(policy_weights,
                                                 transitions.observation)
            lps = action_dists.log_prob(transitions.action)
            return -jnp.sum(lps * advantage_estimates)

        policy_gradient = jax.grad(surrogate_loss)(weights.policy_weights)
        policy_updates, policy_optimizer_state = self._policy_optimizer.update(
            policy_gradient, weights.policy_optimizer_state)
        return PolicyGradientAgentWeights(
            policy_weights=optax.apply_updates(weights.policy_weights,
                                               policy_updates),
            policy_optimizer_state=policy_optimizer_state)


def episode_reward(rewards: jnp.ndarray, done: jnp.ndarray,
                   discount_factor: float):

    size = rewards.shape[0]
    num_future_dones = jnp.cumsum(done[::-1])[::-1]

    relevance_square = num_future_dones == num_future_dones[..., None]
    discounts = (1. / discount_factor) * jnp.cumprod(jnp.where(
        relevance_square, discount_factor, 1.),
                                                     axis=-1) * relevance_square
    return jnp.sum(discounts * rewards[None, ...], axis=-1)