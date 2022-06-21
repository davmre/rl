from absl.testing import parameterized
import tree

import jax
import numpy as np

from daves_rl_lib.internal import type_util
from daves_rl_lib.internal import util


def test_seed(n=1) -> type_util.KeyArray:
    seed = jax.random.PRNGKey(42)
    if n > 1:
        return jax.random.split(seed, n)
    return seed


class TestCase(parameterized.TestCase):

    def _assertNested(self, assert_fn, a, b):

        def _assert_with_path(path, x, y):
            try:
                assert_fn(x, y)
            except Exception as e:
                raise AssertionError(
                    f'Assertion failed at nested structure path: {path}') from e

        util.map_with_debug_paths(_assert_with_path, a, b)

    def assertSameShape(self, a, b):
        self.assertEqual(a.shape, b.shape)

    def assertSameShapeNested(self, a, b):
        self._assertNested(self.assertSameShape, a, b)

    def assertShapeNested(self, a, shape):
        self._assertNested(lambda x, s: self.assertEqual(x.shape, tuple(s)), a,
                           shape)

    def assertAllClose(self, a, b, **kwargs):
        np.testing.assert_allclose(np.float32(a), np.float32(b), **kwargs)

    def assertAllCloseNested(self, a, b, **kwargs):
        self._assertNested(lambda x, y: self.assertAllClose(x, y, **kwargs), a,
                           b)

    def assertAllEqual(self, a, b):
        np.testing.assert_equal(np.asarray(a), np.asarray(b))

    def assertAllEqualNested(self, a, b):
        self._assertNested(self.assertAllEqual, a, b)
