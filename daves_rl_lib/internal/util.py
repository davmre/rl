import collections
from typing import Optional, Union

import tree

import jax
from jax import numpy as jnp
import numpy as np

from daves_rl_lib.internal import type_util


def tree_where(c, a_tree, b_tree):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(c, a, b), a_tree,
                                  b_tree)


def batch_multiply(a, b):
    return jax.vmap(lambda x, y: x * y)(a, b)


def as_numpy_seed(seed):
    if seed is None:
        return seed
    return int(
        jax.random.randint(seed,
                           shape=(),
                           minval=0,
                           maxval=np.iinfo(np.int32).max))


def as_jax_seed(seed: Optional[Union[int, np.ndarray]]) -> type_util.KeyArray:
    if seed is None:
        seed = np.random.randint(low=0, high=np.iinfo(np.int32).max)
    if hasattr(seed, 'shape') and seed.shape == (2,):  # type: ignore
        return jnp.asarray(seed)
    return jax.random.PRNGKey(seed)  # type: ignore


SummaryStats = collections.namedtuple('SummaryStats',
                                      ['mean', 'std', 'median', 'min', 'max'])


def summarize(x):
    return SummaryStats(mean=jnp.mean(x),
                        std=jnp.std(x),
                        median=jnp.median(x),
                        min=jnp.min(x),
                        max=jnp.max(x))


def format(x, fmt=lambda x: np.array_str(np.asarray(x), precision=3)):
    return jax.tree_util.tree_map(fmt, x)


def format_summary(x, **kwargs):
    return format(jax.tree_util.tree_map(summarize, x), **kwargs)


def map_with_debug_paths(fn, *xs):
    # TODO: rewrite using private utils under `jax._src.tree_util`
    # to work with general Pytree registrations.

    def _map_with_paths_helper(path, *xs):
        x1 = xs[0]
        if jax.tree_util.tree_leaves(x1) == [x1]:  # At a leaf.
            return fn(path, *xs)
        if tree.is_nested(x1):
            return tree.map_structure_with_path(
                lambda p, *ys: _map_with_paths_helper(path + p, *ys), *xs)
        if hasattr(x1, '__dataclass_fields__'):
            field_names = x1.__dataclass_fields__.keys()
            mapped = {}
            for field_name in field_names:
                mapped[field_name] = _map_with_paths_helper(
                    path + (field_name,),
                    *tuple(getattr(x, field_name) for x in xs))
            return type(x1)(**mapped)
        if x1 is None:
            return None
        else:
            raise TypeError(
                f"I don't know how to define paths for pytree type {type(x1)}")

    return _map_with_paths_helper((), *xs)