import dataclasses
from os import environ
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from flax import linen
from flax import struct
import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import networks
from daves_rl_lib.algorithms import agent_lib
from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util
from daves_rl_lib.internal import util


@struct.dataclass
class ImplicitQuantileWeights:
    quantile_weights: Any
    quantile_target_weights: Any
    quantile_optimizer_state: Any
    num_steps: jnp.ndarray


class ImplicitQuantileNetwork(linen.Module):
    """Implicit quantile network module."""
    state_embedding_network: linen.Module
    layer_sizes: Sequence[int]
    quantile_embedding_size: int = 64
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @linen.compact
    def __call__(self, state: jnp.ndarray,
                 quantile: jnp.ndarray) -> jnp.ndarray:
        x = self.state_embedding_network(state)
        quantile_embedding = jnp.cos(quantile * jnp.pi *
                                     jnp.arange(self.quantile_embedding_size))
        x_dim = x.shape[-1]
        quantile_projection_layer = linen.Dense(x_dim,
                                                name=f'quantile_projection',
                                                kernel_init=self.kernel_init,
                                                use_bias=self.bias)
        x *= self.activation(quantile_projection_layer(quantile_embedding))
        for i, hidden_size in enumerate(self.layer_sizes):
            layer = linen.Dense(hidden_size,
                                name=f'hidden_{i}',
                                kernel_init=self.kernel_init,
                                use_bias=self.bias)
            x = layer(x)
            if i != len(self.layer_sizes) - 1:
                x = self.activation(x)
        return x

    def apply_at_quantiles(self, w, obs, qs) -> jnp.ndarray:
        return jax.vmap(lambda q: self.apply(w, obs, q))(qs)  # type: ignore


class ImplicitQuantileAgent(agent_lib.ReplayBufferAgent):

    def __init__(self,
                 quantile_net: ImplicitQuantileNetwork,
                 quantile_optimizer: optax.GradientTransformation,
                 num_quantile_samples: int,
                 num_target_return_samples: int,
                 replay_buffer_size: int,
                 epsilon: Union[float, optax.Schedule],
                 target_weights_decay: float,
                 gradient_batch_size: int,
                 discount_factor: float = 1.,
                 entropy_regularization: Optional[float] = None,
                 use_double_estimator: bool = True):
        self._quantile_net = quantile_net
        self._quantile_optimizer = quantile_optimizer
        self._num_quantile_samples = num_quantile_samples
        self._num_target_return_samples = num_target_return_samples
        self._epsilon_fn = epsilon if callable(
            epsilon) else optax.constant_schedule(epsilon)
        self._target_weights_decay = target_weights_decay
        self._discount_factor = discount_factor
        self._entropy_regularization = entropy_regularization
        self._use_double_estimator = use_double_estimator
        super().__init__(replay_buffer_size=replay_buffer_size,
                         gradient_batch_size=gradient_batch_size)

    @property
    def quantile_net(self):
        return self._quantile_net

    def _init_weights(self, seed: type_util.KeyArray,
                      dummy_observation: jnp.ndarray,
                      **kwargs) -> ImplicitQuantileWeights:
        quantile_weights = self.quantile_net.init(seed, dummy_observation, 0.5)
        return ImplicitQuantileWeights(
            quantile_weights=quantile_weights,
            quantile_target_weights=quantile_weights,
            quantile_optimizer_state=self._quantile_optimizer.init(
                quantile_weights),
            num_steps=jnp.zeros([], dtype=jnp.int32))

    def _action_dist(self,
                     weights: ImplicitQuantileWeights,
                     observation: jnp.ndarray,
                     num_quantiles=64) -> tfp.distributions.Distribution:
        quantiles = jnp.linspace(0., 1., num_quantiles)
        qvalues = self.quantile_net.apply_at_quantiles(weights.quantile_weights,
                                                       observation, quantiles)
        best_actions = select_greedy_action(qvalues)
        num_actions = qvalues.shape[-1]
        greedy_dist = tfp.distributions.Categorical(probs=jax.nn.one_hot(
            best_actions, num_classes=num_actions, axis=-1))

        # Epsilon-greedy exploration.
        epsilon = self._epsilon_fn(weights.num_steps)
        logits = greedy_dist.logits_parameter()
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=[1. - epsilon, epsilon]),
            components_distribution=tfp.distributions.Categorical(
                logits=jnp.stack([logits, jnp.zeros_like(logits)], axis=-2)))

    def _update(self, weights: ImplicitQuantileWeights,
                transitions: environment_lib.Transition,
                seed: type_util.KeyArray) -> ImplicitQuantileWeights:

        quantile_loss_fn = build_minibatch_loss_fn(
            build_single_transition_loss_fn(
                self.quantile_net,
                weights.quantile_target_weights,
                discount_factor=self._discount_factor,
                num_target_return_samples=self._num_target_return_samples,
                num_quantile_samples=self._num_quantile_samples), transitions,
            seed)

        # Compute TD error and update the network accordingly.
        quantile_weights_grad, _ = jax.grad(quantile_loss_fn, has_aux=True)(
            weights.quantile_weights)

        (quantile_weights_update,
         quantile_optimizer_state) = self._quantile_optimizer.update(
             quantile_weights_grad, weights.quantile_optimizer_state)
        quantile_weights = optax.apply_updates(weights.quantile_weights,
                                               quantile_weights_update)

        return ImplicitQuantileWeights(
            quantile_weights=quantile_weights,
            quantile_optimizer_state=quantile_optimizer_state,
            # Update the target network as a moving average.
            quantile_target_weights=jax.tree_util.tree_map(
                lambda x, y: self._target_weights_decay * x +
                (1 - self._target_weights_decay) * y,
                weights.quantile_target_weights, quantile_weights),
            num_steps=weights.num_steps + 1)


def build_single_transition_loss_fn(quantile_net, quantile_target_weights: Any,
                                    discount_factor: float,
                                    num_target_return_samples: int,
                                    num_quantile_samples: int):

    def loss_fn(quantile_weights, transition, seed):
        quantiles_seed, target_seed, seed = jax.random.split(seed, 3)
        target_return_quantiles = jax.random.uniform(
            target_seed, shape=[num_target_return_samples])
        quantiles = jax.random.uniform(quantiles_seed,
                                       shape=[num_quantile_samples])

        # shape [num_target_return_samples, num_actions]
        target_qvalue_samples = quantile_net.apply_at_quantiles(
            quantile_target_weights, transition.next_observation,
            target_return_quantiles)
        best_action = select_greedy_action(target_qvalue_samples)
        greedy_target_qvalue_samples = target_qvalue_samples[..., best_action]
        target_return_samples = (
            transition.reward + discount_factor *
            jnp.where(transition.done, 0, greedy_target_qvalue_samples))

        values_at_quantiles = quantile_net.apply_at_quantiles(
            quantile_weights, transition.observation,
            quantiles)[..., transition.action]
        td_errors = (target_return_samples[:, jnp.newaxis] -
                     values_at_quantiles[jnp.newaxis, :])

        aux_return_values = (target_qvalue_samples, target_return_samples,
                             quantiles, values_at_quantiles, td_errors)
        return jnp.mean(quantile_loss(td_errors, quantiles)), aux_return_values

    return loss_fn


def build_minibatch_loss_fn(single_transition_loss_fn, transitions, seed):
    seeds = jax.random.split(seed, transitions.batch_shape[-1])

    def minibatch_loss(w):
        f = lambda t, s: single_transition_loss_fn(w, transition=t, seed=s)
        losses, aux = jax.vmap(f)(transitions, seeds)
        return jnp.mean(losses), aux

    return minibatch_loss


def select_greedy_action(qvalue_samples):
    expected_qvalues = jnp.mean(qvalue_samples, axis=0)
    return jnp.argmax(expected_qvalues, axis=-1)


def huber_loss(x, kappa=1.0):
    return jnp.where(
        jnp.abs(x) <= kappa, 0.5 * jnp.square(x),
        kappa * jnp.abs(x) - 0.5 * kappa * kappa)


def quantile_loss(error, quantile, kappa=1.0):
    """Loss minimized when `error < 0` with probability `quantile`."""
    return (jnp.abs(quantile - jnp.where(error < 0, 1., 0.)) *
            huber_loss(error, kappa) / kappa)
