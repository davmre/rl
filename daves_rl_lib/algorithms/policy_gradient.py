import dataclasses
from email import policy
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from flax import struct
import optax

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


class PolicyGradientAgent(agent_lib.EpisodicAgent):

    def __init__(self,
                 policy_net: networks.FeedForwardModel,
                 policy_optimizer: optax.GradientTransformation,
                 max_num_steps: int,
                 discount_factor: float = 1.):
        self._policy_net = policy_net
        self._policy_optimizer = policy_optimizer
        self._discount_factor = discount_factor
        super().__init__(max_num_steps=max_num_steps)

    @property
    def policy_net(self):
        return self._policy_net

    def _init_weights(self, seed: type_util.KeyArray,
                      dummy_observation: jnp.ndarray,
                      dummy_action: jnp.ndarray):
        del dummy_observation  # Unused.
        del dummy_action  # Unused.
        policy_weights = self.policy_net.init(seed)
        policy_optimizer_state = self._policy_optimizer.init(policy_weights)
        return PolicyGradientAgentWeights(
            policy_weights=policy_weights,
            policy_optimizer_state=policy_optimizer_state)

    def _action_dist(self, obs, weights: PolicyGradientAgentWeights):
        return self.policy_net.apply(weights.policy_weights, obs)

    def _update_episode(
            self, weights: PolicyGradientAgentWeights,
            transitions: environment_lib.Transition,
            num_valid_transitions: jnp.ndarray) -> PolicyGradientAgentWeights:
        value_estimates = rewards_to_go(
            agent_lib.zero_invalid(transitions.reward, num_valid_transitions),
            self._discount_factor)

        def surrogate_loss(policy_weights):
            action_dists = self.policy_net.apply(policy_weights,
                                                 transitions.observation)
            lps = agent_lib.zero_invalid(
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


def rewards_to_go(rewards, discount_factor):
    size = rewards.shape[0]
    discount_square = jnp.array(
        [[discount_factor**(j - i) if j >= i else 0.0
          for j in range(size)]
         for i in range(size)])
    return jnp.sum(rewards[None, ...] * discount_square, axis=-1)
