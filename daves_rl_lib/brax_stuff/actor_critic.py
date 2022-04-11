from typing import Optional
import jax
import jax.numpy as jnp

import optax

from daves_rl_lib import networks
from daves_rl_lib import train

def make_take_action(env,
                     value_net,
                     value_weights,
                     policy_net,
                     policy_weights,
                     action_dist_statistics_fn=lambda d: d.entropy()):
    
    def take_action(state, seed):
        seed, next_seed = jax.random.split(seed, 2)

        def act(w):
            action_dist = policy_net.apply(w, state.obs)
            action_dist_statistics = action_dist_statistics_fn(action_dist)
            action, action_lp = action_dist.experimental_sample_and_log_prob(
                seed=seed)
            return jnp.sum(action_lp), (action, action_dist_statistics)

        (_, (action, action_dist_stats)), score = jax.value_and_grad(
             act, has_aux=True)(policy_weights)

        next_state = jax.tree_util.tree_map(
            lambda x, y: jnp.where(state.done, x, y),
            state,
            env.step(state, action))
        
        next_state_value, next_state_value_grad = jax.value_and_grad(
            lambda w, x: value_net.apply(w, x)[0])(value_weights, next_state.obs)

        next_state_value, next_state_value_grad = jax.tree_util.tree_map(
            lambda v: jnp.where(next_state.done, 0, v),
            (next_state_value, next_state_value_grad))

        return (action,
                action_dist_stats,
                next_state,
                next_state_value,
                next_state_value_grad,
                score,
                next_seed)

    return take_action

def actor_critic_step(env,
                      learner,
                      step: int,
                      policy_net: networks.FeedForwardModel,
                      value_net: networks.FeedForwardModel,
                      policy_optimizer,
                      value_optimizer,
                      policy_trace_decay_rate=0.4,
                      value_trace_decay_rate=0.4,
                      discount_factor=0.97,
                      action_dist_statistics_fn=lambda d: d.entropy()):
    
    take_action = make_take_action(
        env=env,
        value_net=value_net,
        value_weights=learner.value_weights,
        policy_net=policy_net,
        policy_weights=learner.policy_weights,
        action_dist_statistics_fn=action_dist_statistics_fn)
  
    (actions, action_dist_statistics, next_states, next_state_values,
     next_state_value_grads, scores, next_seeds) = jax.vmap(
        take_action)(learner.states, learner.seeds)
    
    temporal_difference_errors = jnp.where(
        learner.states.done,
        0.,
        (next_states.reward +
         discount_factor * next_state_values -
         learner.state_values))

    def average_negative_gradients(trace):
        return jnp.mean(jax.vmap(lambda x, y: -x * y)(
            trace, temporal_difference_errors), axis=0)

    # Update policy trace and weights
    policy_discount = discount_factor ** step
    policy_trace = jax.tree_util.tree_map(
        lambda e, g: policy_trace_decay_rate * discount_factor * e + policy_discount * g,
        learner.policy_trace,
        scores)
    policy_updates, policy_optimizer_state = policy_optimizer.update(
        jax.tree_util.tree_map(average_negative_gradients, policy_trace),
        learner.policy_optimizer_state)
    policy_weights = optax.apply_updates(learner.policy_weights, policy_updates)

    # Update value trace and weights. Note that we update
    # the weights using the *existing* trace before updating the trace itself
    # to incorporate the new value gradient. This implements a TD update
    # of the value of the pre-action state, using the information gained by
    # taking the action.
    value_updates, value_optimizer_state = value_optimizer.update(
        jax.tree_util.tree_map(average_negative_gradients, learner.value_trace),
        learner.value_optimizer_state)
    value_weights = optax.apply_updates(learner.value_weights, value_updates)
        
    value_trace = jax.tree_util.tree_map(
        lambda e, g: value_trace_decay_rate * discount_factor * e + g,
        learner.value_trace,
        next_state_value_grads)
    
    diagnostics = {
        'temporal_difference_error': temporal_difference_errors,
        'action_dist_statistics': action_dist_statistics
    }

    return (actions,
            train.Learner(
              states=next_states,
              state_values=next_state_values,
              value_weights=value_weights,
              value_trace=value_trace,
              value_optimizer_state=value_optimizer_state,
              policy_weights=policy_weights,
              policy_trace=policy_trace,
              policy_optimizer_state=policy_optimizer_state,
              seeds=next_seeds),
            diagnostics)