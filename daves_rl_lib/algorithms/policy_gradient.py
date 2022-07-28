import dataclasses
from typing import Any, Callable, Optional

import jax
from jax.experimental import host_callback
import jax.numpy as jnp

from flax import struct
import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import networks
from daves_rl_lib.algorithms import agent_lib
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util


@struct.dataclass
class PolicyGradientAgentWeights:
    policy_weights: type_util.PyTree
    policy_optimizer_state: type_util.PyTree


class EpisodicPolicyGradientAgent(agent_lib.EpisodicAgent):

    def __init__(self,
                 policy_net: networks.FeedForwardModel,
                 policy_optimizer: optax.GradientTransformation,
                 max_num_steps: int,
                 discount_factor: float = 1.,
                 reward_to_go: bool = True,
                 standardize_advantages: bool = True,
                 ppo_clip_epsilon=0.2,
                 ppo_steps_per_iteration=1):
        self._policy_net = policy_net
        self._policy_optimizer = policy_optimizer
        self._discount_factor = discount_factor
        self._reward_to_go = reward_to_go
        self._standardize_advantages = standardize_advantages
        self._ppo_clip_epsilon = ppo_clip_epsilon
        self._ppo_steps_per_iteration = ppo_steps_per_iteration
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
        surrogate_loss = ppo_surrogate_loss(
            self.policy_net,
            batch_obs=transitions.observation,
            batch_actions=transitions.action,
            batch_advantages=value_estimates,
            num_valid_transitions=num_valid_transitions,
            old_weights=(weights.policy_weights
                         if self._ppo_steps_per_iteration > 1 else None),
            ppo_clip_epsilon=self._ppo_clip_epsilon,
            jit_compile=False)

        policy_weights = weights.policy_weights
        policy_optimizer_state = weights.policy_optimizer_state
        for ppo_step in range(self._ppo_steps_per_iteration):
            policy_updates, policy_optimizer_state = (
                self._policy_optimizer.update(
                    jax.grad(surrogate_loss)(policy_weights),
                    policy_optimizer_state))
            policy_weights = optax.apply_updates(weights.policy_weights,
                                                 policy_updates)
        return PolicyGradientAgentWeights(
            policy_weights=policy_weights,
            policy_optimizer_state=policy_optimizer_state)


class PolicyGradientAgent(agent_lib.PeriodicUpdateAgent):

    def __init__(self,
                 policy_net: networks.FeedForwardModel,
                 policy_optimizer: optax.GradientTransformation,
                 steps_per_update: int,
                 discount_factor: float = 1.,
                 reward_to_go: bool = True,
                 standardize_advantages: bool = True,
                 ppo_clip_epsilon=0.2,
                 ppo_steps_per_iteration=1):
        self._policy_net = policy_net
        self._policy_optimizer = policy_optimizer
        self._discount_factor = discount_factor
        self._reward_to_go = reward_to_go
        self._standardize_advantages = standardize_advantages
        self._ppo_clip_epsilon = ppo_clip_epsilon
        self._ppo_steps_per_iteration = ppo_steps_per_iteration
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
        surrogate_loss = ppo_surrogate_loss(
            self.policy_net,
            batch_obs=transitions.observation,
            batch_actions=transitions.action,
            batch_advantages=self._estimate_advantages(weights=weights,
                                                       transitions=transitions),
            old_weights=(weights.policy_weights
                         if self._ppo_steps_per_iteration > 1 else None),
            ppo_clip_epsilon=self._ppo_clip_epsilon)

        policy_weights = weights.policy_weights
        policy_optimizer_state = weights.policy_optimizer_state
        for _ in range(self._ppo_steps_per_iteration):
            policy_updates, policy_optimizer_state = (
                self._policy_optimizer.update(
                    jax.grad(surrogate_loss)(policy_weights),
                    policy_optimizer_state))
            policy_weights = optax.apply_updates(weights.policy_weights,
                                                 policy_updates)
        return PolicyGradientAgentWeights(
            policy_weights=policy_weights,
            policy_optimizer_state=policy_optimizer_state)


def episode_reward(rewards: jnp.ndarray, done: jnp.ndarray,
                   discount_factor: float):
    num_future_dones = jnp.cumsum(done[::-1])[::-1]

    relevance_square = num_future_dones == num_future_dones[..., None]
    discounts = (1. / discount_factor) * jnp.cumprod(jnp.where(
        relevance_square, discount_factor, 1.),
                                                     axis=-1) * relevance_square
    return jnp.sum(discounts * rewards[None, ...], axis=-1)


def ppo_surrogate_loss(policy_net,
                       batch_obs,
                       batch_actions,
                       batch_advantages,
                       num_valid_transitions=None,
                       old_weights=None,
                       ppo_clip_epsilon=0.2,
                       jit_compile=True):
    """
    Computes the surrogate objective whose autodiff derivative is the policy
    gradient.
    """

    # Prevent gradients to the old weights, in case the user passes
    # old_weights=weights.
    old_weights = jax.tree_util.tree_map(jax.lax.stop_gradient, old_weights)

    def fn_of_weights(w):

        def scaled_lp(adv, action, obs):

            action_lp = policy_net.apply(w, obs).log_prob(action)
            if old_weights is None:
                action_lp_old = jax.lax.stop_gradient(action_lp)
            else:
                action_lp_old = policy_net.apply(old_weights,
                                                 obs).log_prob(action)

            importance_weight = jnp.exp(action_lp - action_lp_old)
            clipped_importance_weight = jnp.clip(importance_weight,
                                                 1 - ppo_clip_epsilon,
                                                 1 + ppo_clip_epsilon)

            lps = jnp.minimum(adv * importance_weight,
                              adv * clipped_importance_weight)
            return lps

        scaled_lps = jax.vmap(scaled_lp)(batch_advantages, batch_actions,
                                         batch_obs)
        if num_valid_transitions is not None:
            scaled_lps = agent_lib.zero_out_suffix_of_elements(
                scaled_lps, num_valid_transitions)
            # Preserve information about the number of transitions in the
            # magnitude of the returned gradients. (makes the mean equivalent
            # to a sum over valid transitions).
            scaled_lps *= num_valid_transitions
        # Return the negative objective as a loss to be minimized.
        return -jnp.mean(scaled_lps)

    return jax.jit(fn_of_weights) if jit_compile else fn_of_weights
