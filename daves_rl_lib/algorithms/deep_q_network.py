import dataclasses
from os import environ
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp

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
class DQNWeights:
    qvalue_weights: Any
    qvalue_target_weights: Any
    qvalue_optimizer_state: Any
    num_steps: jnp.ndarray


class DQNAgent(agent_lib.ReplayBufferAgent):

    def __init__(self,
                 qvalue_net: networks.FeedForwardModel,
                 qvalue_optimizer: optax.GradientTransformation,
                 replay_buffer_size: int,
                 epsilon: Union[float, optax.Schedule],
                 target_weights_decay: float,
                 gradient_batch_size: int,
                 discount_factor: float = 1.,
                 entropy_regularization: Optional[float] = None,
                 use_double_estimator: bool = True):
        self._qvalue_net = qvalue_net
        self._qvalue_optimizer = qvalue_optimizer
        self._epsilon_fn = epsilon if callable(
            epsilon) else optax.constant_schedule(epsilon)
        self._target_weights_decay = target_weights_decay
        self._discount_factor = discount_factor
        self._entropy_regularization = entropy_regularization
        self._use_double_estimator = use_double_estimator
        super().__init__(replay_buffer_size=replay_buffer_size,
                         gradient_batch_size=gradient_batch_size)

    @property
    def qvalue_net(self):
        return self._qvalue_net

    def _init_weights(self, seed: type_util.KeyArray, **kwargs) -> DQNWeights:
        qvalue_weights = self.qvalue_net.init(seed)
        return DQNWeights(
            qvalue_weights=qvalue_weights,
            qvalue_target_weights=qvalue_weights,
            qvalue_optimizer_state=self._qvalue_optimizer.init(qvalue_weights),
            num_steps=jnp.zeros([], dtype=jnp.int32))

    def _action_dist(
            self, weights: DQNWeights,
            observation: jnp.ndarray) -> tfp.distributions.Distribution:
        qvalues = self.qvalue_net.apply(weights.qvalue_weights, observation)
        num_actions = qvalues.shape[-1]
        if self._entropy_regularization is None:
            greedy_dist = tfp.distributions.Categorical(probs=jax.nn.one_hot(
                jnp.argmax(qvalues, axis=-1), num_classes=num_actions, axis=-1))
        else:
            greedy_dist = tfp.distributions.Categorical(
                logits=qvalues / self._entropy_regularization)

        # Epsilon-greedy exploration.
        epsilon = self._epsilon_fn(weights.num_steps)
        logits = greedy_dist.logits_parameter()
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=[1. - epsilon, epsilon]),
            components_distribution=tfp.distributions.Categorical(
                logits=jnp.stack([logits, jnp.zeros_like(logits)], axis=-2)))

    def _update(self, weights: DQNWeights,
                transitions: environment_lib.Transition,
                seed: type_util.KeyArray) -> DQNWeights:

        td_error = temporal_difference_loss_fn(
            transitions=transitions,
            qvalue_net=self.qvalue_net,
            qvalue_target_weights=weights.qvalue_target_weights,
            entropy_regularization=self._entropy_regularization,
            discount_factor=self._discount_factor,
            use_double_estimator=self._use_double_estimator)
        # Compute TD error and update the network accordingly.
        qvalue_weights_grad, _ = jax.grad(td_error,
                                          has_aux=True)(weights.qvalue_weights)

        (qvalue_weights_update,
         qvalue_optimizer_state) = self._qvalue_optimizer.update(
             qvalue_weights_grad, weights.qvalue_optimizer_state)
        qvalue_weights = optax.apply_updates(weights.qvalue_weights,
                                             qvalue_weights_update)

        return DQNWeights(
            qvalue_weights=qvalue_weights,
            qvalue_optimizer_state=qvalue_optimizer_state,
            # Update the target network as a moving average.
            qvalue_target_weights=jax.tree_util.tree_map(
                lambda x, y: self._target_weights_decay * x +
                (1 - self._target_weights_decay) * y,
                weights.qvalue_target_weights, qvalue_weights),
            num_steps=weights.num_steps + 1)


def optimal_state_value(observation: jnp.ndarray,
                        qvalue_net: networks.FeedForwardModel,
                        qvalue_target_weights: type_util.PyTree,
                        qvalue_selection_weights: Optional[type_util.PyTree]):
    qvalues = qvalue_net.apply(qvalue_target_weights, observation)
    if qvalue_selection_weights is not None:
        selection_qvalues = qvalue_net.apply(qvalue_selection_weights,
                                             observation)
        selected_action = jnp.argmax(selection_qvalues, axis=-1, keepdims=True)
        return util.batch_gather(qvalues, selected_action)[..., 0]
    return jnp.max(qvalues, axis=-1)


def optimal_regularized_state_value(
        observation: jnp.ndarray, qvalue_net: networks.FeedForwardModel,
        qvalue_target_weights: type_util.PyTree,
        qvalue_selection_weights: Optional[type_util.PyTree],
        entropy_regularization: float):
    qvalues = qvalue_net.apply(qvalue_target_weights, observation)
    if qvalue_selection_weights is not None:
        selection_qvalues = qvalue_net.apply(qvalue_selection_weights,
                                             observation)
        naive_softmax_value = entropy_regularization * jsp.special.logsumexp(
            selection_qvalues / entropy_regularization, axis=-1)
        selection_probs = jax.nn.softmax(selection_qvalues, axis=-1)
        correction = jnp.sum(selection_probs * (qvalues - selection_qvalues),
                             axis=-1)
        return naive_softmax_value - correction
    return entropy_regularization * jsp.special.logsumexp(
        qvalues / entropy_regularization, axis=-1)


def qvalues_and_td_error(transition: environment_lib.Transition,
                         qvalue_net: networks.FeedForwardModel,
                         qvalue_weights: type_util.PyTree,
                         qvalue_target_weights: type_util.PyTree,
                         use_double_estimator: bool = True,
                         discount_factor=1.,
                         entropy_regularization: Optional[float] = None):
    """

    Args:

    Returns:
      qvalues: (batch of) scalar action values estimated for the transitions.
      target_values: (batch  of) scalar values estimated for the post-transition
        states.
      td_error: (batch of) scalar temporal difference error(s).
    """

    qvalue_selection_weights = (jax.lax.stop_gradient(qvalue_weights)
                                if use_double_estimator else None)
    if entropy_regularization is None:
        next_state_values = optimal_state_value(
            observation=transition.next_observation,
            qvalue_net=qvalue_net,
            qvalue_target_weights=qvalue_target_weights,
            qvalue_selection_weights=qvalue_selection_weights)
    else:
        next_state_values = optimal_regularized_state_value(
            observation=transition.next_observation,
            qvalue_net=qvalue_net,
            qvalue_target_weights=qvalue_target_weights,
            qvalue_selection_weights=qvalue_selection_weights,
            entropy_regularization=entropy_regularization)
    target_values = transition.reward + jnp.where(
        transition.done, 0, discount_factor * next_state_values)
    qvalues = jnp.take_along_axis(qvalue_net.apply(qvalue_weights,
                                                   transition.observation),
                                  transition.action[..., None],
                                  axis=-1)[..., 0]
    return qvalues, target_values, qvalues - target_values


def temporal_difference_loss_fn(transitions: environment_lib.Transition,
                                qvalue_net: networks.FeedForwardModel,
                                qvalue_target_weights,
                                discount_factor: float = 1.,
                                use_double_estimator: bool = True,
                                entropy_regularization: Optional[float] = None):
    """Temporal difference objective as a function of the Q value weights."""

    def mean_squared_td_error(weights):
        qvalues, target_values, td_errors = qvalues_and_td_error(
            transition=transitions,
            qvalue_net=qvalue_net,
            qvalue_weights=weights,
            qvalue_target_weights=qvalue_target_weights,
            entropy_regularization=entropy_regularization,
            use_double_estimator=use_double_estimator,
            discount_factor=discount_factor)
        return 0.5 * jnp.mean(td_errors**2, axis=-1), (qvalues, target_values)

    return mean_squared_td_error
