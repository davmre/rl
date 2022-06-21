from typing import Any

from absl.testing import parameterized

import numpy as np
import jax
from jax import numpy as jnp

from flax import struct

from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import test_util
from daves_rl_lib import util


@struct.dataclass
class DummyStruct:
    a: Any
    b: Any


class UtilTest(test_util.TestCase):

    def test_map_with_debug_paths(self):
        x = [3., {'inner_dict': DummyStruct(a=[2, 3], b={'c': 4})}]
        paths = util.map_with_debug_paths(lambda p, x: p, x)
        expected_paths = [(0,), (1, 'inner_dict', 'a', 0),
                          (1, 'inner_dict', 'a', 1),
                          (1, 'inner_dict', 'b', 'c')]

        x_td = jax.tree_util.tree_structure(x)
        self.assertEqual(x_td.flatten_up_to(paths), expected_paths)
