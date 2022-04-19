import collections

import numpy as np

import jax
from jax import numpy as jnp


def tree_where(c, a_tree, b_tree):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(c, a, b), a_tree,
                                  b_tree)


def batch_multiply(a, b):
    return jax.vmap(lambda x, y: x * y)(a, b)


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