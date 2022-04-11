
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
    policy_net, value_net, learner = train.initialize_learner(
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

    return env, learner, step_fn, policy_net, value_net



class ActorCriticTests(parameterized.TestCase):
    
    def assertSameShape(self, a, b):
        self.assertEqual(a.shape, b.shape)

    def assertSameShapeNested(self, a, b):
        jax.tree_util.tree_map(self.assertSameShape, a, b)
    
    def test_step_preserves_shapes(self):
        batch_size = 32
        env, learner, step_fn, _, _ = setup_test_environment(
            seed=jax.random.PRNGKey(0),
            batch_size=batch_size)
        action, new_learner, diagnostics = step_fn(learner=learner, step=0)
        self.assertSameShapeNested(learner, new_learner)

    def test_learns_in_trivial_environment(self):
        batch_size = 64
        discount_factor = 0.97
        env, learner, step_fn, _, value_net = setup_test_environment(
            seed=jax.random.PRNGKey(0),
            batch_size=batch_size,
            discount_factor=discount_factor)
        jitted_epoch = jax.jit(
            functools.partial(
                train.advance_epoch,
                env=env,
                value_net=value_net,
                step_fn=step_fn,
                max_num_steps=20,
                discount_factor=discount_factor))
        for epoch in range(150):
            learner, diagnostics = jitted_epoch(learner=learner)
            diagnostics = jax.tree_util.tree_map(float, diagnostics)
            if (epoch + 1) % 10 == 0:
                print("Epoch:", epoch, diagnostics)
            
        self.assertGreaterEqual(
            diagnostics['initial_value'].mean, 0.95)
        self.assertGreaterEqual(
            diagnostics['return'].mean, 0.96)
        self.assertLessEqual(
            diagnostics['trajectory_length'].mean, 2.1)

if __name__ == '__main__':
    absltest.main()