from typing import Optional
import jax
import jax.numpy as jnp

import optax

from daves_rl_lib import networks

def make_take_action(env, value_net, value_weights, policy_net, policy_weights):
    def take_action(state, seed):
        seed, next_seed = jax.random.split(seed, 2)

        def act(w):
            action_dist = policy_net.apply(w, state.obs)
            action, action_lp = action_dist.experimental_sample_and_log_prob(seed=seed)
            return jnp.sum(action_lp), action

        (action_lp, action), score = jax.value_and_grad(act, has_aux=True)(policy_weights)

        next_state = jax.tree_util.tree_map(
            lambda x, y: jnp.where(state.done, x, y),
            state,
            env.step(state, action))
        
        next_state_value, next_state_value_grad = jax.value_and_grad(
            lambda w, x: value_net.apply(w, x)[0])(value_weights, next_state.obs)

        next_state_value, next_state_value_grad = jax.tree_util.tree_map(
            lambda v: jnp.where(next_state.done, 0, v),
            (next_state_value, next_state_value_grad))

        return action, next_state, next_state_value, next_state_value_grad, score, next_seed

    return take_action

def actor_critic_step(env,
                      states,
                      state_values: jnp.ndarray,
                      step: int,
                      policy_net: networks.FeedForwardModel,
                      value_net: networks.FeedForwardModel,
                      policy_weights,
                      policy_trace,
                      value_weights,
                      value_trace,
                      policy_optimizer,
                      policy_optimizer_state,
                      value_optimizer,
                      value_optimizer_state,
                      seeds: jax.random.PRNGKey,
                      value_trace_decay_rate=0.4,
                      policy_trace_decay_rate=0.4,
                      discount_factor=0.97):
    
    take_action = make_take_action(env=env,
                                   value_net=value_net,
                                   value_weights=value_weights,
                                   policy_net=policy_net,
                                   policy_weights=policy_weights)
  
    actions, next_states, next_state_values, next_state_value_grads, scores, next_seeds = jax.vmap(
        take_action)(states, seeds)
    
    temporal_difference_errors = jnp.where(
        states.done,
        0.,
        next_states.reward + discount_factor * next_state_values - state_values)
    
    print("shapes: reward {} value {} next value {} td {}".format(
        next_states.reward.shape, state_values.shape, next_state_values.shape, temporal_difference_errors.shape))
    
    def average_negative_gradients(trace):
        return jnp.mean(jax.vmap(lambda x, y: -x * y)(trace, temporal_difference_errors), axis=0)

    # Update policy trace and weights
    policy_discount = discount_factor ** step
    policy_trace = jax.tree_util.tree_map(
        lambda e, g: policy_trace_decay_rate * discount_factor * e + policy_discount * g,
        policy_trace,
        scores)
    policy_updates, policy_optimizer_state = policy_optimizer.update(
        jax.tree_util.tree_map(average_negative_gradients, policy_trace),
        policy_optimizer_state)
    policy_weights = optax.apply_updates(policy_weights, policy_updates)

    # Update value trace and weights. Note that we update
    # the weights using the *existing* trace before updating the trace itself
    # to incorporate the new value gradient. This implements a TD update
    # of the value of the pre-action state, using the information gained by
    # taking the action.
    value_updates, value_optimizer_state = value_optimizer.update(
        jax.tree_util.tree_map(average_negative_gradients, value_trace),
        value_optimizer_state)
    value_weights = optax.apply_updates(value_weights, value_updates)
        
    value_trace = jax.tree_util.tree_map(
        lambda e, g: value_trace_decay_rate * discount_factor * e + g,
        value_trace,
        next_state_value_grads)

    return (actions,
            next_states, next_state_values,
            value_weights, value_trace, value_optimizer_state,
            policy_weights, policy_trace, policy_optimizer_state,
            next_seeds)