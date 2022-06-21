import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from flax import linen
from flax import struct
from tensorflow_probability.substrates import jax as tfp


@dataclasses.dataclass
class FeedForwardModel:
    init: Any
    apply: Any


@struct.dataclass
class NetworkInTraining:
    weights: Any
    optimizer_state: Any


class MLP(linen.Module):
    """MLP module."""
    layer_sizes: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    activate_final: Optional[Callable] = None
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        x = data
        for i, hidden_size in enumerate(self.layer_sizes):
            layer = linen.Dense(hidden_size,
                                name=f'hidden_{i}',
                                kernel_init=self.kernel_init,
                                use_bias=self.bias)
            x = layer(x)
            if i != len(self.layer_sizes) - 1:
                x = self.activation(x)
        if self.activate_final:
            x = self.activate_final(x)
        return x


def from_logits(dist_cls,
                logits,
                reinterpreted_batch_ndims=None,
                event_shape_from_logits_size=lambda n: [n / 2]):
    event_shape = event_shape_from_logits_size(logits.shape[-1])
    logits_idx = 0
    params = {}
    for param, properties in dist_cls.parameter_properties().items():
        if not properties.is_preferred:
            continue
        param_shape = properties.shape_fn(event_shape)
        bij = properties.default_constraining_bijector_fn()
        logits_size = bij.inverse_event_shape_tensor(param_shape)[-1]
        params[param] = bij.forward(logits[...,
                                           logits_idx:logits_idx + logits_size])
    dist = dist_cls(**params)
    if reinterpreted_batch_ndims is not None:
        dist = tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
    return dist


def categorical_from_logits(logits):
    return tfp.distributions.Categorical(logits=logits)


def make_model(layer_sizes: Sequence[int],
               obs_size: int,
               activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
               activate_final=None):
    dummy_obs = jnp.ones([obs_size])
    module = MLP(layer_sizes=layer_sizes,
                 activation=activation,
                 activate_final=activate_final)
    return FeedForwardModel(init=lambda rng: module.init(rng, dummy_obs),
                            apply=module.apply)
