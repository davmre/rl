import dataclasses
from typing import Any, Callable, Optional

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
class A2CWeights:
    policy_weights: Any
    policy_optimizer_state: Any
    value_weights: Any
    value_optimizer_state: Any


class A2CAgent(agent_lib.PeriodicUpdateAgent):

    def __init__(
            self,
            policy_net,  # Should include final logits.
            value_net,  # Should include final [1] output.
            policy_optimizer,
            value_optimizer,
            steps_per_update: int,
            entropy_regularization: float = 0.01,
            discount_factor=1.):
        self._policy_net = policy_net
        self._value_net = value_net
        self._policy_optimizer = policy_optimizer
        self._value_optimizer = value_optimizer
        self._entropy_regularization = entropy_regularization
        self._discount_factor = discount_factor
        super().__init__(steps_per_update=steps_per_update)

    @property
    def policy_net(self):
        return self._policy_net

    @property
    def value_net(self):
        return self._value_net

    def _init_weights(self, seed: type_util.KeyArray) -> A2CWeights:
        policy_seed, value_seed = jax.random.split(seed, 2)
        policy_weights = self._policy_net.init(policy_seed)
        value_weights = self._value_net.init(value_seed)
        return A2CWeights(
            value_weights=value_weights,
            value_optimizer_state=self._value_optimizer.init(value_weights),
            policy_weights=policy_weights,
            policy_optimizer_state=self._policy_optimizer.init(policy_weights))

    def _action_dist(self, weights: A2CWeights,
                     observation: jnp.ndarray) -> jnp.ndarray:
        return self._policy_net.apply(weights.policy_weights, observation)

    def _estimate_qvalues(self, value_weights: Any,
                          transitions: environment_lib.Transition):
        final_state_value = self.value_net.apply(
            value_weights, transitions.next_observation[-1])[0]
        return agent_lib.episode_reward_to_go(
            rewards=transitions.reward,
            done=transitions.done,
            discount_factor=self._discount_factor,
            final_state_value=final_state_value)

    def _estimate_advantage(self, value_weights: Any,
                            transitions: environment_lib.Transition):
        state_values = self.value_net.apply(value_weights,
                                            transitions.observation)[..., 0]
        # The empirical q-values form 'targets' for the value function, so treat
        # the value weights as fixed in that context (TODO: should we have a
        # separate set of target weights?).
        return self._estimate_qvalues(jax.lax.stop_gradient(value_weights),
                                      transitions) - state_values

    def _single_worker_gradients(self, transitions: environment_lib.Transition,
                                 policy_weights, value_weights):

        def value_net_loss(w):
            advantages = self._estimate_advantage(w, transitions)
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
        return value_grad, policy_grad

    def _update(self, weights: A2CWeights,
                transitions: environment_lib.Transition) -> A2CWeights:
        # Compute and aggregate gradients across workers.
        compute_gradients = lambda transitions: self._single_worker_gradients(
            transitions=transitions,
            policy_weights=weights.policy_weights,
            value_weights=weights.value_weights)
        if len(transitions.done.shape) > 1:
            compute_gradients = wrap_to_return_mean(jax.vmap(compute_gradients))

        value_grad, policy_grad = compute_gradients(transitions)
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
                          value_optimizer_state=value_optimizer_state)


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


def wrap_to_return_mean(f):
    return lambda *a, **kw: jax.tree_util.tree_map(
        lambda x: jnp.mean(x, axis=0), f(*a, **kw))
