import dataclasses
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from flax import struct
import optax

from daves_rl_lib import networks
from daves_rl_lib.algorithms import exploration_lib
from daves_rl_lib.algorithms import replay_buffer
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.internal import type_util
from daves_rl_lib.internal import util


@struct.dataclass
class DeepQLearner:
    agent_states: environment_lib.State
    replay_buffer: replay_buffer.ReplayBuffer
    qvalue_weights: Any
    qvalue_target_weights: Any
    qvalue_optimizer_state: Any
    seed: type_util.KeyArray


def initialize_learner(env: environment_lib.Environment,
                       qvalue_net: networks.FeedForwardModel,
                       qvalue_optimizer,
                       buffer_size: int,
                       seed: type_util.KeyArray,
                       batch_size: Optional[int] = None):
    episodes_seed, weights_seed, buffer_seed = jax.random.split(seed, 3)
    initial_weights = qvalue_net.init(weights_seed)
    return DeepQLearner(
        agent_states=env.reset(batch_size=batch_size, seed=episodes_seed),
        replay_buffer=replay_buffer.ReplayBuffer.initialize_empty(
            size=buffer_size,
            dummy_state=env.reset(seed=buffer_seed),
            action_shape=env.action_space.shape),
        qvalue_weights=initial_weights,
        qvalue_target_weights=jax.tree_util.tree_map(jnp.zeros_like,
                                                     initial_weights),
        qvalue_optimizer_state=qvalue_optimizer.init(initial_weights),
        seed=buffer_seed)


def qvalues_and_td_error(state,
                         action,
                         next_state,
                         qvalue_net,
                         qvalue_weights,
                         qvalue_target_weights,
                         discount_factor=1.):
    """

    Args:
      state: structure of arrays with optional batch dimension.
      
    Returns:
      qvalues: (batch of) scalar action values estimated for the transitions.
      target_values: (batch  of) scalar values estmated for the post-transition states.
      td_error: (batch of) scalar temporal difference error(s).
    """
    next_state_values = jnp.where(
        next_state.done, 0,
        jnp.max(qvalue_net.apply(qvalue_target_weights, next_state.observation),
                axis=-1))
    target_values = next_state.reward + discount_factor * next_state_values
    qvalues = jnp.take_along_axis(qvalue_net.apply(qvalue_weights,
                                                   state.observation),
                                  action[..., None],
                                  axis=-1)[..., 0]
    return qvalues, target_values, qvalues - target_values


def qvalue_weights_grad_from_transitions(transitions: replay_buffer.Transition,
                                         qvalue_net: networks.FeedForwardModel,
                                         qvalue_weights, qvalue_target_weights,
                                         discount_factor):
    """Computes the gradient of the loss function w.r.t. the weights."""

    def squared_td_error(weights):
        qvalues, target_values, td_errors = qvalues_and_td_error(
            state=transitions.state,
            action=transitions.action,
            next_state=transitions.next_state,
            qvalue_net=qvalue_net,
            qvalue_weights=weights,
            qvalue_target_weights=qvalue_target_weights,
            discount_factor=discount_factor)
        return 0.5 * jnp.sum(td_errors**2, axis=-1), (qvalues, target_values)

    grad, (qvalues, target_values) = jax.grad(squared_td_error,
                                              has_aux=True)(qvalue_weights)
    return grad, qvalues, target_values


def update_qvalue_network(learner: DeepQLearner,
                          qvalue_net: networks.FeedForwardModel,
                          qvalue_optimizer, gradient_batch_size: int,
                          target_weights_decay: float,
                          discount_factor: float) -> DeepQLearner:
    """Trains the Q network using a batch of transitions from the buffer."""
    seed, next_seed = jax.random.split(learner.seed)
    # Compute TD error and update the network accordingly.
    qvalue_weights_grad, _, _ = qvalue_weights_grad_from_transitions(
        transitions=learner.replay_buffer.sample_uniform(
            batch_shape=(gradient_batch_size,), seed=seed),
        qvalue_net=qvalue_net,
        qvalue_weights=learner.qvalue_weights,
        qvalue_target_weights=learner.qvalue_target_weights,
        discount_factor=discount_factor)
    qvalue_weights_update, qvalue_optimizer_state = qvalue_optimizer.update(
        qvalue_weights_grad, learner.qvalue_optimizer_state)
    qvalue_weights = optax.apply_updates(learner.qvalue_weights,
                                         qvalue_weights_update)

    # Update the target network as a moving average.
    qvalue_target_weights = jax.tree_util.tree_map(
        lambda x, y: target_weights_decay * x + (1 - target_weights_decay) * y,
        learner.qvalue_target_weights, qvalue_weights)

    return DeepQLearner(agent_states=learner.agent_states,
                        replay_buffer=learner.replay_buffer,
                        qvalue_weights=qvalue_weights,
                        qvalue_optimizer_state=qvalue_optimizer_state,
                        qvalue_target_weights=qvalue_target_weights,
                        seed=next_seed)


def collect_and_buffer_jax_transitions(learner: DeepQLearner,
                                       env: environment_lib.Environment,
                                       qvalue_net: networks.FeedForwardModel,
                                       epsilon: float) -> DeepQLearner:
    seed, next_seed = jax.random.split(learner.seed, 2)
    step_seeds = jax.random.split(seed, learner.agent_states.seed.shape[-2])

    def step(state, step_seed):
        state = env.reset_if_done(state)
        action = exploration_lib.select_action(
            state.observation,
            policy_fn=exploration_lib.epsilon_greedy_policy(
                qvalue_net=qvalue_net,
                qvalue_weights=learner.qvalue_weights,
                epsilon=epsilon),
            seed=step_seed)
        next_state = env.step(state, action)
        return replay_buffer.Transition(
            state, action, next_state,
            qvalues_and_td_error(
                state=state,
                action=action,
                next_state=next_state,
                qvalue_net=qvalue_net,
                qvalue_weights=learner.qvalue_weights,
                qvalue_target_weights=learner.qvalue_target_weights,
                discount_factor=env.discount_factor)[-1])

    transitions = jax.vmap(step)(learner.agent_states, step_seeds)
    return dataclasses.replace(
        learner,
        agent_states=transitions.next_state,
        replay_buffer=learner.replay_buffer.with_transitions(transitions),
        seed=next_seed)


def deep_q_update_step(learner: DeepQLearner, env: environment_lib.Environment,
                       qvalue_net: networks.FeedForwardModel, qvalue_optimizer,
                       gradient_batch_size: int, target_weights_decay: float,
                       epsilon: float) -> DeepQLearner:
    # take a step in the (batch of) environments and collect transition(s)
    # in the replay buffer
    learner = collect_and_buffer_jax_transitions(env=env,
                                                 learner=learner,
                                                 qvalue_net=qvalue_net,
                                                 epsilon=epsilon)

    # sample a (batch of) transition(s) from the buffer and update the network
    return update_qvalue_network(learner=learner,
                                 qvalue_net=qvalue_net,
                                 qvalue_optimizer=qvalue_optimizer,
                                 gradient_batch_size=gradient_batch_size,
                                 discount_factor=env.discount_factor,
                                 target_weights_decay=target_weights_decay)


def compile_deep_q_update_step_stateful(
        env: environment_lib.ExternalEnvironment,
        qvalue_net: networks.FeedForwardModel,
        qvalue_optimizer,
        gradient_batch_size: int,
        target_weights_decay: float,
        epsilon: float,
        jit_compile=True) -> Callable[[DeepQLearner], DeepQLearner]:
    jit = jax.jit if jit_compile else lambda f: f

    select_action = jax.jit(
        lambda obs, weights, seed: exploration_lib.select_action(
            obs,
            seed=seed,
            policy_fn=exploration_lib.epsilon_greedy_policy(
                qvalue_net=qvalue_net, qvalue_weights=weights, epsilon=epsilon))
    )
    buffer_transition = jax.jit(lambda rb, t: rb.with_transition(t))
    update_network = jax.jit(lambda l: update_qvalue_network(
        learner=l,
        qvalue_net=qvalue_net,
        qvalue_optimizer=qvalue_optimizer,
        gradient_batch_size=gradient_batch_size,
        discount_factor=env.discount_factor,
        target_weights_decay=target_weights_decay))

    def update_learner(learner: DeepQLearner) -> DeepQLearner:
        seed, next_seed = jax.random.split(learner.seed)
        state = learner.agent_states
        if state.done:
            state = env.reset()
        action = select_action(state.observation,
                               weights=learner.qvalue_weights,
                               seed=seed)
        next_state = env.step(action)
        updated_replay_buffer = buffer_transition(
            learner.replay_buffer,
            replay_buffer.Transition(state, action, next_state, td_error=0.))
        learner = dataclasses.replace(learner,
                                      agent_states=next_state,
                                      replay_buffer=updated_replay_buffer,
                                      seed=next_seed)
        return update_network(learner)

    return update_learner
