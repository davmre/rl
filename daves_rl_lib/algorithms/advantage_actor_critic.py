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
    steps_buffer: replay_buffer.ReplayBuffer


class A2CAgent(agent_lib.Agent):

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
        self._steps_per_update = steps_per_update
        self._entropy_regularization = entropy_regularization
        self._discount_factor = discount_factor
        super().__init__()

    @property
    def policy_net(self):
        return self._policy_net

    @property
    def value_net(self):
        return self._value_net

    def _init_weights(self, seed: type_util.KeyArray,
                      dummy_observation: jnp.ndarray, dummy_action: jnp.ndarray,
                      batch_size: int) -> A2CWeights:
        policy_seed, value_seed, state_seed = jax.random.split(seed, 3)
        policy_weights = self._policy_net.init(policy_seed)
        value_weights = self._value_net.init(value_seed)
        return A2CWeights(
            value_weights=value_weights,
            value_optimizer_state=self._value_optimizer.init(value_weights),
            policy_weights=policy_weights,
            policy_optimizer_state=self._policy_optimizer.init(policy_weights),
            steps_buffer=jax.vmap(
                lambda _: replay_buffer.ReplayBuffer.initialize_empty(
                    size=self._steps_per_update,
                    observation=dummy_observation,
                    action=dummy_action))(jnp.arange(batch_size)))

    def _action_dist(self, weights: A2CWeights,
                     observation: jnp.ndarray) -> jnp.ndarray:
        return self._policy_net.apply(weights.policy_weights, observation)

    def _update(self, weights: A2CWeights,
                transition: environment_lib.Transition):
        """_summary_

        Args:
            weights: _description_
            transition: batch of transitions from acting in a batch of states.
        """
        weights = dataclasses.replace(
            weights,
            steps_buffer=jax.vmap(lambda b, t: b.with_transition(t))(
                weights.steps_buffer, transition))
        return jax.lax.cond(
            # Formally check if any worker's buffer is full, though since all
            # buffers are the same size they must fill (and be reset)
            # at the same time.
            jnp.any(weights.steps_buffer.is_full),
            lambda: self._update_weights_and_reset_buffer(weights),
            lambda: weights)

    def _update_weights_and_reset_buffer(self,
                                         weights: A2CWeights) -> A2CWeights:
        # Compute and aggregate gradients across workers.
        value_grads, policy_grads = jax.vmap(
            lambda transitions: self._single_worker_gradients(
                transitions=transitions,
                policy_weights=weights.policy_weights,
                value_weights=weights.value_weights))(
                    weights.steps_buffer.transitions)
        value_grad, policy_grad = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), (value_grads, policy_grads))

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
                          steps_buffer=jax.vmap(lambda b: b.reset())(
                              weights.steps_buffer))

    def _single_worker_gradients(self, transitions: environment_lib.Transition,
                                 policy_weights, value_weights):
        state_values, values_grad = jax.vmap(lambda s: jax.value_and_grad(
            lambda w: self.value_net.apply(w, s)[0])(value_weights))(
                transitions.observation)

        final_state_value = self.value_net.apply(
            value_weights, transitions.next_observation[-1])[0]

        estimated_returns = episode_reward_to_go(
            reward=jnp.concatenate(
                [transitions.reward,
                 jnp.array([final_state_value])]),
            done=jnp.concatenate([transitions.done,
                                  jnp.array([False])]),
            discount_factor=self._discount_factor)[:-1]

        advantage = estimated_returns - state_values

        # Value and policy gradients are computed at a per-step scale, so that
        # increasing `num_steps` does *not* increase the scale of the gradients;
        # it just reduces their variance. It's not clear to me if this makes
        # sense - typically when we increase `num_steps` per update we
        # decrease the number of outer-loop updates, and debatably we want
        # the 'total learning' to remain constant under that adjustment, which
        # would suggest increasing the scale. However, the current approach is
        # most consistent with the style convention that batch size parameters
        # (which `num_steps` sort-of is) are independent of learning rates.
        value_grad = jax.tree_util.tree_map(
            (lambda x: jnp.mean(util.batch_multiply(-advantage, x), axis=0)),
            values_grad)

        policy_grad = batch_policy_gradient(self.policy_net,
                                            policy_weights,
                                            batch_obs=transitions.observation,
                                            batch_actions=transitions.action,
                                            batch_advantages=advantage)
        policy_entropy, entropy_grad = jax.value_and_grad(lambda w: jnp.mean(
            self.policy_net.apply(w, transitions.observation).entropy()))(
                policy_weights)
        regularized_policy_grad = jax.tree_util.tree_map(
            lambda a, b: a + self._entropy_regularization * b, policy_grad,
            entropy_grad)
        return value_grad, policy_grad


def episode_reward_to_go(reward, done, discount_factor):
    size = reward.shape[0]
    num_future_dones = jnp.cumsum(done[::-1])[::-1]
    episode_discount_square = jnp.array([[
        discount_factor**(j - i) *
        (num_future_dones[i] == num_future_dones[j]) if j >= i else 0.0
        for j in range(size)
    ]
                                         for i in range(size)])
    return jnp.sum(episode_discount_square * reward[None, ...], axis=-1)


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