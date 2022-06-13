import dataclasses

import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp


def select_action(observation, policy_fn, seed):
    action_dist = policy_fn(observation)
    if len(action_dist.batch_shape):
        raise ValueError(
            'A single input produced a batch policy: {}. You may need to '
            'wrap the output distribution with `tfd.Independent`'.format(
                action_dist))
    return action_dist.sample(seed=seed)


def epsilon_greedy_policy(qvalue_net, qvalue_weights, epsilon=1e-2):

    def policy_fn(obs):
        qvalues = qvalue_net.apply(qvalue_weights, obs)
        batch_shape, num_actions = qvalues.shape[:-1], qvalues.shape[-1]
        greedy_actions = jnp.argmax(qvalues, axis=-1, keepdims=True)
        probs = jnp.ones_like(qvalues) * epsilon / (num_actions - 1)
        if num_actions == 1:
            probs = jnp.ones_like(qvalues)
        else:
            probs = jnp.where(
                jnp.arange(num_actions) == greedy_actions, 1 - epsilon,
                epsilon / (num_actions - 1))
        return tfp.distributions.Categorical(probs=probs)

    return policy_fn