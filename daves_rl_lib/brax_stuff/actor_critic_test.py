
import functools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
from jax import numpy as jnp

import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import train

from daves_rl_lib.brax_stuff import actor_critic
from daves_rl_lib.brax_stuff import trivial_environment


def setup_test_environment(seed, env_size=2, env_dim=1, batch_size=32, discount_factor = 0.97):
    
    env = trivial_environment.TargetEnv(size=env_size, dim=env_dim)
    policy_optimizer = optax.adam(1e-3)
    value_optimizer = optax.adam(1e-3)
    (
        (state, state_value, state_seeds),
        (policy_net, policy_weights, policy_optimizer_state),
        (value_net, value_weights, value_optimizer_state)
    ) = train.initialize_learner(
        env,
        policy_layer_sizes=[32, env._dim * 2],
        value_layer_sizes=[256, 32, 1],
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_activate_final=tfp.distributions.Categorical,
        batch_size=batch_size,
        seed=seed)

    step_fn = functools.partial(
        actor_critic.actor_critic_step,
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        discount_factor=discount_factor)

    return (
        (env, state, state_value, state_seeds),
        (policy_weights, policy_optimizer_state),
        (value_weights, value_optimizer_state),
        step_fn)



class ActorCriticTests(parameterized.TestCase):
    
    def assertSameShape(self, a, b):
        self.assertEqual(a.shape, b.shape)

    def assertSameShapeNested(self, a, b):
        jax.tree_util.tree_map(self.assertSameShape, a, b)
    
    def test_step_preserves_shapes(self):
        batch_size = 32
        (
            (_, state, state_value, state_seeds),
            (policy_weights, policy_optimizer_state),
            (value_weights, value_optimizer_state),
            step_fn
        ) = setup_test_environment(seed=jax.random.PRNGKey(0),
                                   batch_size=batch_size)
        policy_trace, value_trace = jax.tree_util.tree_map(
            lambda x: jnp.zeros((batch_size,) + x.shape),
            (policy_weights, value_weights))
            
        (action,
         next_state,
         next_state_value,
         new_value_weights,
         new_value_trace,
         new_value_optimizer_state,
         new_policy_weights,
         new_policy_trace,
         new_policy_optimizer_state,
         new_seeds) = (
            step_fn(
                states=state,
                state_values=state_value,
                step=0,
                policy_weights=policy_weights,
                policy_trace=policy_trace,
                policy_optimizer_state=policy_optimizer_state,
                value_weights=value_weights,
                value_trace=value_trace,
                value_optimizer_state=value_optimizer_state,
                seeds=state_seeds))
        
        self.assertSameShapeNested(state, next_state)
        self.assertSameShapeNested(state_value, next_state_value)

        self.assertSameShapeNested(policy_weights, new_policy_weights)
        self.assertSameShapeNested(value_weights, new_value_weights)

        self.assertSameShapeNested(policy_trace, new_policy_trace)
        self.assertSameShapeNested(value_trace, new_value_trace)

        self.assertSameShapeNested(policy_optimizer_state, new_policy_optimizer_state)
        self.assertSameShapeNested(value_optimizer_state, new_value_optimizer_state)
        
        self.assertSameShape(state_seeds, new_seeds)

    def test_learns_in_trivial_environment(self):
        batch_size = 64
        discount_factor = 0.97
        (
            (env, initial_state, initial_state_value, seeds),
            (policy_weights, policy_optimizer_state),
            (value_weights, value_optimizer_state),
            step_fn,
        ) = setup_test_environment(seed=jax.random.PRNGKey(0),
                                   batch_size=batch_size,
                                   discount_factor=discount_factor)

        jitted_step = jax.jit(step_fn)
        average_returns = []
        average_steps = []
        for epoch in range(300):
            (rollout,
             seeds,
             policy_weights,
             policy_optimizer_state,
             value_weights,
             value_optimizer_state) = train.rollout_trajectory(
                env=env,
                initial_state=initial_state,
                initial_state_value=initial_state_value,
                step_fn=jitted_step,
                max_num_steps=30,
                seeds=seeds,
                policy_weights=policy_weights,
                policy_optimizer_state=policy_optimizer_state,
                value_weights=value_weights,
                value_optimizer_state=value_optimizer_state)

            total_steps = rollout[-1].step
            total_return = train.discounted_return(rollout, discount_factor)
            average_returns.append(jnp.mean(total_return))
            average_steps.append(jnp.mean(total_steps))
            print("epoch {} average per-step return {} average steps {}".format(
                epoch, average_returns[-1], average_steps[-1]))
        self.assertLessEqual(np.mean(average_steps[-10:]), 2.2)

if __name__ == '__main__':
    absltest.main()