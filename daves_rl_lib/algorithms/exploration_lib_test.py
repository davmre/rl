from absl.testing import parameterized

import numpy as np
import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib.internal import test_util
from daves_rl_lib import networks

from daves_rl_lib.algorithms import exploration_lib


class ExplorationTests(test_util.TestCase):

    @parameterized.named_parameters([('_no_batch', np.asarray([1., 2.]), 4),
                                     ('_batch',
                                      np.asarray([[1., 2.], [3., 4.],
                                                  [5., 6.]]), 4),
                                     ('_one_action', np.array([1., 2.]), 1)])
    def test_epsilon_greedy(self, obs, num_actions):
        batch_shape = obs.shape[:-1]

        net = networks.make_model([num_actions], obs_size=obs.shape[-1])
        weights = net.init(test_util.test_seed())

        qvalues = net.apply(weights, obs)
        self.assertEqual(qvalues.shape, batch_shape + (num_actions,))
        best_actions = jnp.argmax(qvalues, axis=-1)

        epsilon = 1e-1
        policy_fn = exploration_lib.epsilon_greedy_policy(net,
                                                          weights,
                                                          epsilon=epsilon)
        action_dist = policy_fn(obs)
        self.assertEqual(action_dist.batch_shape, batch_shape)
        self.assertEqual(action_dist.event_shape, ())
        probs_of_best_actions = jnp.take_along_axis(action_dist.probs,
                                                    best_actions[..., None],
                                                    axis=-1)[..., 0]
        if num_actions > 1:
            self.assertAllClose(probs_of_best_actions,
                                jnp.ones(batch_shape) * (1 - epsilon))
        else:
            self.assertAllClose(probs_of_best_actions, 1.)
