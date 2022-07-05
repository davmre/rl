import dataclasses
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from flax import struct
import optax

from daves_rl_lib import networks
from daves_rl_lib.algorithms import agent_lib
from daves_rl_lib.algorithms import exploration_lib
from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util
from daves_rl_lib.internal import util


@struct.dataclass
class A2CAuxiliaryQuantities:
    policy_entropies: jnp.ndarray
    advantages: jnp.ndarray
    policy_grad_norm: jnp.ndarray
    value_grad_norm: jnp.ndarray
    entropy_grad_norm: jnp.ndarray


@struct.dataclass
class A2CWeights:
    policy_weights: Any
    policy_optimizer_state: Any
    value_weights: Any
    value_optimizer_state: Any
    value_target_weights: Any
    auxiliary: Optional[A2CAuxiliaryQuantities] = None


class A2CAgent(agent_lib.PeriodicUpdateAgent):

    def __init__(
            self,
            policy_net,  # Should include final logits.
            value_net,  # Should include final [1] output.
            policy_optimizer,
            value_optimizer,
            steps_per_update: int,
            entropy_regularization: float = 0.01,
            target_weights_decay: float = 0.9,
            discount_factor=1.,
            standardize_advantages: bool = False,
            keep_auxiliary=False):
        self._policy_net = policy_net
        self._value_net = value_net
        self._policy_optimizer = policy_optimizer
        self._value_optimizer = value_optimizer
        self._entropy_regularization = entropy_regularization
        self._target_weights_decay = target_weights_decay
        self._discount_factor = discount_factor
        self._standardize_advantages = standardize_advantages
        self._keep_auxiliary = keep_auxiliary
        super().__init__(steps_per_update=steps_per_update)

    @property
    def policy_net(self):
        return self._policy_net

    @property
    def value_net(self):
        return self._value_net

    def _init_weights(self,
                      seed: type_util.KeyArray,
                      batch_size: Optional[int] = None) -> A2CWeights:
        policy_seed, value_seed = jax.random.split(seed, 2)
        policy_weights = self._policy_net.init(policy_seed)
        value_weights = self._value_net.init(value_seed)
        auxiliary = None
        if self._keep_auxiliary:
            batch_shape = (batch_size,) if batch_size else ()
            auxiliary = A2CAuxiliaryQuantities(
                policy_entropies=jnp.zeros(batch_shape, dtype=jnp.float32),
                advantages=jnp.zeros(batch_shape + (self._steps_per_update,),
                                     dtype=jnp.float32),
                policy_grad_norm=jnp.zeros(batch_shape),
                entropy_grad_norm=jnp.zeros(batch_shape),
                value_grad_norm=jnp.zeros(batch_shape))
        return A2CWeights(
            value_weights=value_weights,
            value_optimizer_state=self._value_optimizer.init(value_weights),
            value_target_weights=value_weights,
            policy_weights=policy_weights,
            policy_optimizer_state=self._policy_optimizer.init(policy_weights),
            auxiliary=auxiliary)

    def _action_dist(self, weights: A2CWeights,
                     observation: jnp.ndarray) -> jnp.ndarray:
        return self._policy_net.apply(weights.policy_weights, observation)

    def _estimate_qvalues(self, value_target_weights: Any,
                          transitions: environment_lib.Transition):
        final_state_value = self.value_net.apply(
            value_target_weights, transitions.next_observation[-1])[0]
        return agent_lib.episode_reward_to_go(
            rewards=transitions.reward,
            done=transitions.done,
            discount_factor=self._discount_factor,
            final_state_value=final_state_value)

    def _estimate_advantage(self, value_weights: Any, value_target_weights: Any,
                            transitions: environment_lib.Transition):
        state_values = self.value_net.apply(value_weights,
                                            transitions.observation)[..., 0]
        # The empirical q-values form 'targets' for the value function, so treat
        # the value weights as fixed in that context (TODO: should we have a
        # separate set of target weights?).
        advantages = (
            self._estimate_qvalues(value_target_weights, transitions) -
            state_values)
        if self._standardize_advantages:
            advantages_ = jax.lax.stop_gradient(advantages)
            advantages = (advantages -
                          jnp.mean(advantages_)) / (jnp.std(advantages_) + 1e-6)
        return advantages

    def _single_worker_gradients(self, transitions: environment_lib.Transition,
                                 policy_weights, value_weights,
                                 value_target_weights):

        def value_net_loss(w):
            advantages = self._estimate_advantage(
                w,
                value_target_weights=value_target_weights,
                transitions=transitions)
            squared_loss = 0.5 * jnp.mean(advantages**2, axis=-1)
            return squared_loss, advantages

        value_grad, advantages = jax.grad(value_net_loss,
                                          has_aux=True)(value_weights)

        policy_grad = batch_policy_gradient(self.policy_net,
                                            policy_weights,
                                            batch_obs=transitions.observation,
                                            batch_actions=transitions.action,
                                            batch_advantages=advantages)
        policy_entropy, entropy_grad = jax.value_and_grad(lambda w: jnp.mean(
            self.policy_net.apply(w, transitions.observation).entropy()))(
                policy_weights)
        regularized_policy_grad = jax.tree_util.tree_map(
            lambda a, b: a + self._entropy_regularization * b, policy_grad,
            entropy_grad)
        return value_grad, regularized_policy_grad, (
            advantages, policy_entropy, agent_lib.tree_norm(policy_grad),
            agent_lib.tree_norm(entropy_grad), agent_lib.tree_norm(value_grad))

    def _update(self, weights: A2CWeights,
                transitions: environment_lib.Transition) -> A2CWeights:
        # Compute and aggregate gradients across workers.
        compute_gradients = lambda transitions: self._single_worker_gradients(
            transitions=transitions,
            policy_weights=weights.policy_weights,
            value_weights=weights.value_weights,
            value_target_weights=weights.value_target_weights)
        if len(transitions.done.shape) > 1:
            value_grads, policy_grads, (
                advantages, policy_entropies, policy_grad_norm,
                entropy_grad_norm,
                value_grad_norm) = jax.vmap(compute_gradients)(transitions)
            value_grad, policy_grad = jax.tree_util.tree_map(
                lambda x: jnp.mean(x, axis=0), (value_grads, policy_grads))
        else:
            value_grad, policy_grad, (
                advantages, policy_entropies, policy_grad_norm,
                entropy_grad_norm,
                value_grad_norm) = compute_gradients(transitions)

        value_updates, value_optimizer_state = self._value_optimizer.update(
            value_grad, weights.value_optimizer_state)
        value_weights = optax.apply_updates(weights.value_weights,
                                            value_updates)
        policy_updates, policy_optimizer_state = self._policy_optimizer.update(
            # We want to *maximize* reward, rather than minimize loss, so
            # pass negative gradients to the optimizer.
            jax.tree_util.tree_map(lambda g: -g, policy_grad),
            weights.policy_optimizer_state)
        policy_weights = optax.apply_updates(weights.policy_weights,
                                             policy_updates)

        return A2CWeights(policy_weights=policy_weights,
                          policy_optimizer_state=policy_optimizer_state,
                          value_weights=value_weights,
                          value_optimizer_state=value_optimizer_state,
                          value_target_weights=agent_lib.update_moving_average(
                              weights.value_target_weights,
                              value_weights,
                              decay=self._target_weights_decay),
                          auxiliary=A2CAuxiliaryQuantities(
                              policy_entropies=policy_entropies,
                              advantages=advantages,
                              policy_grad_norm=policy_grad_norm,
                              entropy_grad_norm=entropy_grad_norm,
                              value_grad_norm=value_grad_norm)
                          if self._keep_auxiliary else weights.auxiliary)


def batch_policy_gradient(policy_net, policy_weights, batch_obs, batch_actions,
                          batch_advantages):
    """
    Computes `mean(advantages * scores, axis=0)` where
    `scores[i] = grad(policy_lp[i], policy_weights)`
    is the batch of score function vectors.
    """

    def fn_of_weights(w):

        def scaled_lp(adv, action, obs):
            return adv * policy_net.apply(w, obs).log_prob(action)

        scaled_lps = jax.vmap(scaled_lp)(batch_advantages, batch_actions,
                                         batch_obs)
        return jnp.mean(scaled_lps)

    return jax.grad(fn_of_weights)(policy_weights)
