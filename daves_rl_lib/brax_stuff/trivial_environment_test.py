import functools

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp

from daves_rl_lib.brax_stuff import trivial_environment


class TrivialEnvironmentTests(parameterized.TestCase):

    def test_one_dim_transitions(self):
        env = trivial_environment.DiscreteTargetEnvironment(size=3, dim=1)
        self.assertEqual(env.action_size, 2)

        state = env.reset(seed=None)
        self.assertSequenceEqual(state.obs, [0])
        # Move left until we bounce off the edge.
        for expected_idx in (-1, -2, -3, -3, -3):
            state = env.step(state, 0)
            self.assertSequenceEqual(state.obs, [expected_idx])
            self.assertFalse(state.done)
            self.assertEqual(state.reward, 0.)

        # Now move right
        for expected_idx in (-2, -1, 0, 1, 2):
            state = env.step(state, 1)
            self.assertSequenceEqual(state.obs, [expected_idx])
            self.assertFalse(state.done)
            self.assertEqual(state.reward, 0.)

        # Transition to terminal state
        state = env.step(state, 1)
        self.assertSequenceEqual(state.obs, [3])
        self.assertTrue(state.done)
        self.assertEqual(state.reward, 1.)

        # Actions in terminal state are no-ops with no reward.
        state = env.step(state, 0)
        self.assertSequenceEqual(state.obs, [3])
        self.assertTrue(state.done)
        self.assertEqual(state.reward, 0.)

    def test_multi_dim_transitions(self):
        env = trivial_environment.DiscreteTargetEnvironment(size=2, dim=2)
        self.assertEqual(env.action_size, 4)

        state = env.reset(seed=None)
        self.assertSequenceEqual(state.obs, (0, 0))
        for (action, expected_obs) in ((1, (0, -1)), (1, (0, -2)), (1, (0, -2)),
                                       (0, (-1, -2)), (2, (0, -2)),
                                       (2, (1, -2)), (2, (2, -2)), (3, (2,
                                                                        -1))):
            state = env.step(state, action)
            self.assertSequenceEqual(state.obs, expected_obs)
            self.assertFalse(state.done)
            self.assertEqual(state.reward, 0.)

        # Transition to terminal state
        state = env.step(state, 3)
        self.assertSequenceEqual(state.obs, (2, 0))
        self.assertTrue(state.done)
        self.assertEqual(state.reward, 1.)

        # Actions in terminal state are no-ops with no reward.
        state = env.step(state, 0)
        self.assertSequenceEqual(state.obs, (2, 0))
        self.assertTrue(state.done)
        self.assertEqual(state.reward, 0.)


if __name__ == '__main__':
    absltest.main()