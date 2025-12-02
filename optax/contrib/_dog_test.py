# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `dog.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax import tree_utils
from optax._src import numerics
from optax.contrib import _dog
from optax._src.numerics import safe_norm


def _setup_parabola(dtype):
  """Quadratic function as a simple test case."""
  initial_params = jnp.array([-1.0, 1.0, 1.0, -1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, -1.0, 1.0], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return jnp.sum((params - final_params) ** 2)

  return initial_params, final_params, get_updates


def _setup_rosenbrock(dtype):
  """Rosenbrock function as a simple test case."""
  a = 1.0
  b = 100.0

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return (a - params[0]) ** 2 + b * (params[1] - params[0] ** 2) ** 2

  return initial_params, final_params, get_updates


class DogTest(chex.TestCase):

  @parameterized.product(
      layer_wise=[True, False],
      dtype=[jnp.float32],
  )
  def test_dog(self, layer_wise, dtype):
    initial_params, final_params, get_updates = _setup_parabola(dtype)
    optimizer = _dog.dog(layer_wise=layer_wise)
    state = optimizer.init(initial_params)

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      updates, state = optimizer.update(updates, state, params)
      params = tree_utils.tree_add(params, updates)
      return params, state

    params = initial_params
    for _ in range(2000):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=1e-2)

  @parameterized.product(
      dtype=[jnp.float32],
  )
  def test_dowg(self, dtype):
    initial_params, final_params, get_updates = _setup_parabola(dtype)
    optimizer = _dog.dowg()
    state = optimizer.init(initial_params)

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      updates, state = optimizer.update(updates, state, params)
      params = tree_utils.tree_add(params, updates)
      return params, state

    params = initial_params
    for _ in range(5000):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
