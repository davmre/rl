import dataclasses
from os import environ
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp

from flax import struct
import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import networks
from daves_rl_lib.algorithms import agent_lib
from daves_rl_lib.algorithms import exploration_lib
from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util
from daves_rl_lib.internal import util


@struct.dataclass
class DQNWeights:
    replay_buffer: replay_buffer.ReplayBuffer
    qvalue_weights: Any
    qvalue_target_weights: Any
    qvalue_optimizer_state: Any
    num_steps: jnp.ndarray
    seed: type_util.KeyArray


class DQNAgent(agent_lib.Agent):

    def __init__(self,
                 qvalue_net: networks.FeedForwardModel,
                 qvalue_optimizer: optax.GradientTransformation,
                 replay_buffer_size: int,
                 epsilon: Union[float, optax.Schedule],
                 target_weights_decay: float,
                 gradient_batch_size: int,
                 discount_factor: float = 1.):
        self._qvalue_net = qvalue_net
        self._qvalue_optimizer = qvalue_optimizer
        self._replay_buffer_size = replay_buffer_size
        self._epsilon_fn = epsilon if callable(
            epsilon) else optax.constant_schedule(epsilon)
        self._target_weights_decay = target_weights_decay
        self._gradient_batch_size = gradient_batch_size
        self._discount_factor = discount_factor
        super().__init__()

    @property
    def qvalue_net(self):
        return self._qvalue_net

    def _init_weights(self, seed: type_util.KeyArray,
                      dummy_observation: jnp.ndarray,
                      dummy_action: jnp.ndarray):
        seed, weights_seed = jax.random.split(seed, 2)
        qvalue_weights = self.qvalue_net.init(weights_seed)
        return DQNWeights(
            replay_buffer=replay_buffer.ReplayBuffer.initialize_empty(
                size=self._replay_buffer_size,
                observation=dummy_observation,
                action=dummy_action),
            qvalue_weights=qvalue_weights,
            qvalue_target_weights=qvalue_weights,
            qvalue_optimizer_state=self._qvalue_optimizer.init(qvalue_weights),
            num_steps=jnp.zeros([], dtype=jnp.int32),
            seed=seed)

    def _action_dist(
            self, weights: DQNWeights,
            observation: jnp.ndarray) -> tfp.distributions.Distribution:
        policy_fn = exploration_lib.epsilon_greedy_policy(
            qvalue_net=self.qvalue_net,
            qvalue_weights=weights.qvalue_weights,
            epsilon=self._epsilon_fn(weights.num_steps))
        return policy_fn(observation)

    def _update(self, weights: DQNWeights,
                transition: environment_lib.Transition) -> DQNWeights:
        batch_shape = transition.done.shape
        if batch_shape:
            replay_buffer = weights.replay_buffer.with_transitions(transition)
        else:
            replay_buffer = weights.replay_buffer.with_transition(transition)

        seed, replay_seed = jax.random.split(weights.seed)
        # Compute TD error and update the network accordingly.
        qvalue_weights_grad, _, _ = qvalue_weights_grad_from_transitions(
            transitions=replay_buffer.sample_uniform(
                batch_shape=(self._gradient_batch_size,), seed=replay_seed),
            qvalue_net=self.qvalue_net,
            qvalue_weights=weights.qvalue_weights,
            qvalue_target_weights=weights.qvalue_target_weights,
            discount_factor=self._discount_factor)

        (qvalue_weights_update,
         qvalue_optimizer_state) = self._qvalue_optimizer.update(
             qvalue_weights_grad, weights.qvalue_optimizer_state)
        qvalue_weights = optax.apply_updates(weights.qvalue_weights,
                                             qvalue_weights_update)

        return DQNWeights(
            replay_buffer=replay_buffer,
            qvalue_weights=qvalue_weights,
            qvalue_optimizer_state=qvalue_optimizer_state,
            # Update the target network as a moving average.
            qvalue_target_weights=jax.tree_util.tree_map(
                lambda x, y: self._target_weights_decay * x +
                (1 - self._target_weights_decay) * y,
                weights.qvalue_target_weights, qvalue_weights),
            seed=seed,
            num_steps=weights.num_steps + 1)


def qvalues_and_td_error(transition: environment_lib.Transition,
                         qvalue_net: networks.FeedForwardModel,
                         qvalue_weights: type_util.PyTree,
                         qvalue_target_weights: type_util.PyTree,
                         discount_factor=1.):
    """

    Args:

    Returns:
      qvalues: (batch of) scalar action values estimated for the transitions.
      target_values: (batch  of) scalar values estimated for the post-transition
        states.
      td_error: (batch of) scalar temporal difference error(s).
    """
    next_state_values = jnp.where(
        transition.done, 0,
        jnp.max(qvalue_net.apply(qvalue_target_weights,
                                 transition.next_observation),
                axis=-1))
    target_values = transition.reward + discount_factor * next_state_values
    qvalues = jnp.take_along_axis(qvalue_net.apply(qvalue_weights,
                                                   transition.observation),
                                  transition.action[..., None],
                                  axis=-1)[..., 0]
    return qvalues, target_values, qvalues - target_values


def qvalue_weights_grad_from_transitions(
        transitions: environment_lib.Transition,
        qvalue_net: networks.FeedForwardModel, qvalue_weights,
        qvalue_target_weights, discount_factor):
    """Computes the gradient of the loss function w.r.t. the weights."""

    def mean_squared_td_error(weights):
        qvalues, target_values, td_errors = qvalues_and_td_error(
            transition=transitions,
            qvalue_net=qvalue_net,
            qvalue_weights=weights,
            qvalue_target_weights=qvalue_target_weights,
            discount_factor=discount_factor)
        return 0.5 * jnp.mean(td_errors**2, axis=-1), (qvalues, target_values)

    grad, (qvalues, target_values) = jax.grad(mean_squared_td_error,
                                              has_aux=True)(qvalue_weights)
    return grad, qvalues, target_values
