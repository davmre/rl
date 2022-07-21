from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import drivers
from daves_rl_lib import networks
from daves_rl_lib.algorithms import policy_gradient
from daves_rl_lib.environments import trivial_environment
from daves_rl_lib.internal import test_util
from daves_rl_lib.internal import util


class EpisodicPolicyGradientTests(test_util.TestCase):

    def test_batch_policy_gradient(self):
        policy_seed, obs_seed, action_seed, advantage_seed = jax.random.split(
            jax.random.PRNGKey(0), num=4)
        num_steps = 17
        policy_net = networks.make_model(
            layer_sizes=[4],
            obs_size=2,
            activate_final=tfp.distributions.Categorical)
        policy_weights = policy_net.init(policy_seed)

        observations = jax.random.normal(obs_seed, shape=(num_steps, 2))
        action_dist = policy_net.apply(policy_weights, observations)
        actions = action_dist.sample(seed=action_seed)
        advantages = jax.random.normal(advantage_seed, shape=(num_steps,))

        policy_loss_fn = policy_gradient.ppo_surrogate_loss(
            policy_net,
            batch_obs=observations,
            batch_actions=actions,
            batch_advantages=advantages)
        policy_grad = jax.grad(policy_loss_fn)(policy_weights)

        # Compare to explicit calculation
        def score(obs, a):
            return jax.grad(lambda w: policy_net.apply(w, obs).log_prob(a))(
                policy_weights)

        expected_policy_grad = jax.tree_util.tree_map(
            lambda s: -jnp.mean(util.batch_multiply(advantages, s), axis=0),
            jax.vmap(score)(observations, actions))

        self.assertAllCloseNested(policy_grad, expected_policy_grad, atol=1e-5)

    @parameterized.named_parameters([('', 1), ('_ppo', 3)])
    def test_learns_in_trivial_discrete_environment(self,
                                                    ppo_steps_per_iteration):

        discount_factor = 0.6
        env = trivial_environment.DiscreteTargetEnvironment(
            size=2,
            dim=1,
            discount_factor=discount_factor,
            one_hot_features=True)

        states = env.reset(seed=test_util.test_seed())
        initial_state_obs = states.observation

        seed = test_util.test_seed()

        agent = policy_gradient.EpisodicPolicyGradientAgent(
            policy_net=networks.make_model(
                [32, env.action_space.num_actions],  # type: ignore
                obs_size=env.observation_size,
                activate_final=networks.categorical_from_logits),
            policy_optimizer=optax.adam(0.1),
            discount_factor=discount_factor,
            max_num_steps=100,
            ppo_clip_epsilon=0.2,
            ppo_steps_per_iteration=ppo_steps_per_iteration)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=states.observation,
            dummy_action=env.action_space.dummy_action())

        step_fn = jax.jit(drivers.jax_driver(env=env, agent=agent))

        done = []
        returns = []
        for step in range(64):
            states, weights, seed = step_fn(states, weights, seed)
            done.append(states.done)
            returns.append(states.episode_return)

        # We should learn a policy that always moves to the right from the
        # start state and its successor state.
        policy_probs = agent.policy_net.apply(
            weights.agent_weights.policy_weights,
            jnp.asarray([
                [0., 0., 1., 0., 0.],  # Start state.
                [0., 0., 0., 1., 0.]  # Successor state.
            ])).probs_parameter()
        self.assertAllClose(policy_probs,
                            jnp.asarray([[0.0, 1.0], [0.0, 1.0]]),
                            atol=0.05)