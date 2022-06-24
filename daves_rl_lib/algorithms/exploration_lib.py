from typing import Union

import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp


def epsilon_greedy_policy(qvalue_net,
                          qvalue_weights,
                          epsilon: Union[float, jnp.ndarray] = 1e-2):

    def policy_fn(obs):
        qvalues = qvalue_net.apply(qvalue_weights, obs)
        batch_shape, num_actions = qvalues.shape[:-1], qvalues.shape[-1]
        greedy_actions = jnp.argmax(qvalues, axis=-1, keepdims=True)
        probs = jnp.ones_like(qvalues) * epsilon / (num_actions - 1)
        if num_actions == 1:
            probs = jnp.ones_like(qvalues)
        else:
            probs = jnp.where(
                jnp.arange(num_actions) == greedy_actions,
                (1 - epsilon), 0.) + epsilon / num_actions
        return tfp.distributions.Categorical(probs=probs)

    return policy_fn