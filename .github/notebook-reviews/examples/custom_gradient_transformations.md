# Creating a custom gradient transformation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/optax/blob/main/examples/custom_gradient_transformations.ipynb)

Optax optimizers are built from small gradient transformations. This tutorial
shows how to write one yourself, give it state, and compose it into an optimizer.
You only need familiarity with Python functions, JAX arrays, and gradients.

We will:

1. Implement a stateless transformation to learn the `init`/`update` interface.
2. Implement momentum with a PyTree state.
3. Combine momentum and a learning rate, then run a JIT-compiled training loop.
4. Check the implementation and extend it to use the current parameters.

Install `optax` if needed (`pip install optax`). The examples can run on CPU
and use only JAX and Optax; no dataset or accelerator is required.

```python
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
```

## The transformation interface

An [`optax.GradientTransformation`](https://optax.readthedocs.io/en/latest/api/generated/optax.GradientTransformation.html#optax.GradientTransformation) contains two functions:

- `init(params)` constructs the initial state from the parameter shapes and
  dtypes. Initialize this state **once**, before training.
- `update(updates, state, params=None)` returns `(transformed_updates, new_state)`.
  Its input is usually a gradient tree. In a chain, it is the output of the
  preceding transformation.

Neither function modifies its arguments in place. The state is a PyTree of JAX
arrays, and the output updates have the same PyTree structure as the input.
The optional `params` argument lets a transformation inspect the current
parameters. We will ignore it initially.

The transformation does not apply its updates to the parameters.
[`optax.apply_updates`](https://optax.readthedocs.io/en/latest/api/generated/optax.apply_updates.html#optax.apply_updates) performs that separate operation by **adding**
the updates. A gradient descent optimizer therefore needs a negative learning
rate multiplier.


## 1. A stateless transformation

First implement multiplication by a fixed scalar $a$:

$$u_t = a g_t.$$

The factory captures the scalar in a closure. There are no evolving statistics,
so `init_fn` returns [`optax.EmptyState`](https://optax.readthedocs.io/en/latest/api/generated/optax.EmptyState.html#optax.EmptyState). Mapping over the tree makes
the same transformation work for a single array, a dictionary, or nested tuples.

```python
def scale_updates(factor: float) -> optax.GradientTransformation:
  """Multiply each update by a fixed scalar.

  Args:
    factor: Scalar multiplier applied to every update leaf.

  Returns:
    A stateless gradient transformation.
  """

  def init_fn(params: optax.Params) -> optax.EmptyState:
    del params
    return optax.EmptyState()

  def update_fn(
      updates: optax.Updates,
      state: optax.EmptyState,
      params: Optional[optax.Params] = None,
  ) -> tuple[optax.Updates, optax.EmptyState]:
    del params
    scaled = jax.tree.map(lambda g: factor * g, updates)
    return scaled, state

  return optax.GradientTransformation(init_fn, update_fn)
```

```python
params = {
    'w': jnp.array([1.0, -2.0], dtype=jnp.float32),
    'b': jnp.array(0.5, dtype=jnp.float32),
}
grads = {
    'w': jnp.array([2.0, -4.0], dtype=jnp.float32),
    'b': jnp.array(1.0, dtype=jnp.float32),
}

tx = scale_updates(-0.1)
state = tx.init(params)
updates, state = tx.update(grads, state, params)
new_params = optax.apply_updates(params, updates)
print(new_params)
```

The parameters become `w = [0.8, -1.6]` and `b = 0.4`: this is one gradient
descent step. Optax already provides [`optax.scale`](https://optax.readthedocs.io/en/latest/api/generated/optax.scale.html#optax.scale); the small
implementation above is useful for understanding the interface.


## 2. A transformation with momentum

Momentum keeps a running trace of the updates. We will use the convention of
[`optax.trace`](https://optax.readthedocs.io/en/latest/api/generated/optax.trace.html#optax.trace) (without Nesterov acceleration):

$$v_0 = 0, \qquad v_{t+1} = \beta v_t + g_t, \qquad u_t = v_{t+1}.$$

Here $0 \leq \beta < 1$. This is an **unnormalized** trace, so there is no
$(1-\beta)$ factor or bias correction. Define the state with a `NamedTuple`;
JAX recognizes it as a PyTree. Its `velocity` field has the same structure,
shapes, and dtypes as the parameters used to initialize it.

```python
class MomentumState(NamedTuple):
  """State containing the running momentum trace."""

  velocity: optax.Updates
```

`init_fn` creates zero velocity. Each call to `update_fn` reads the previous
velocity and returns a **new** state. Hyperparameters such as `decay` live in
the factory closure; evolving arrays live in the state passed through `update`.

The factory checks bounds only for Python scalar hyperparameters. JAX array
values may be tracers, whose values cannot be used in a Python `if` statement.
This guard allows the factory to be called by `optax.inject_hyperparams` inside
JIT-compiled `init` and `update`. When passing an array or a schedule, the caller
is responsible for keeping its values in the documented range.

```python
def scale_by_momentum(
    decay: jax.typing.ArrayLike = 0.9,
) -> optax.GradientTransformation:
  """Accumulate an unnormalized momentum trace.

  Args:
    decay: Scalar momentum coefficient in the interval [0, 1).

  Returns:
    A transformation that outputs the updated momentum trace.

  Raises:
    ValueError: If decay is a Python scalar outside [0, 1).
  """
  if isinstance(decay, (int, float)) and not 0.0 <= decay < 1.0:
    raise ValueError('decay must be in [0, 1).')

  def init_fn(params: optax.Params) -> MomentumState:
    velocity = jax.tree.map(jnp.zeros_like, params)
    return MomentumState(velocity=velocity)

  def update_fn(
      updates: optax.Updates,
      state: MomentumState,
      params: Optional[optax.Params] = None,
  ) -> tuple[optax.Updates, MomentumState]:
    del params
    velocity = jax.tree.map(lambda g, v: decay * v + g, updates, state.velocity)
    return velocity, MomentumState(velocity=velocity)

  return optax.GradientTransformation(init_fn, update_fn)
```

```python
decay = 0.8
momentum = scale_by_momentum(decay=decay)
initial_state = momentum.init(params)
first_updates, first_state = momentum.update(grads, initial_state)
second_updates, second_state = momentum.update(grads, first_state)
print('First trace:', first_updates)
print('Second trace:', second_updates)
```

With the same gradient twice, the first trace is $g$ and the second is $1.8g$.
Reusing `initial_state` at every step would discard momentum: always pass the
returned state into the next call.

For this tutorial we use floating-point parameters and gradients with matching
shapes and explicit dtypes, including scalar leaves. To avoid recompilation
under `jax.jit` (and to be usable inside `jax.lax.scan`), keep the state
structure, shapes, dtypes, and weak types fixed between iterations. An optimizer
for mixed precision may need a separate, explicitly chosen accumulator dtype,
as in `optax.trace`.


## 3. Compose and train with JIT

Our momentum transformation produces a direction. A learning rate transformation
converts it into an additive parameter update:

$$w_{t+1} = w_t - \eta v_{t+1}.$$

[`optax.chain`](https://optax.readthedocs.io/en/latest/api/generated/optax.chain.html#optax.chain) runs transformations in the order listed and manages
their separate states. [`optax.scale_by_learning_rate`](https://optax.readthedocs.io/en/latest/api/generated/optax.scale_by_learning_rate.html#optax.scale_by_learning_rate) supplies the
negative sign and also supports learning rate schedules.

```python
optimizer = optax.chain(
    scale_by_momentum(decay=decay),
    optax.scale_by_learning_rate(0.1),
)

target = {
    'w': jnp.array([1.0, -2.0], dtype=jnp.float32),
    'b': jnp.array(0.5, dtype=jnp.float32),
}
train_params = jax.tree.map(jnp.zeros_like, target)
opt_state = optimizer.init(train_params)


def loss_fn(params: optax.Params) -> jax.Array:
  # This synthetic objective simply pulls the parameters towards target.
  squared_errors = jax.tree.map(
      lambda p, t: jnp.sum((p - t) ** 2), params, target
  )
  return 0.5 * sum(jax.tree.leaves(squared_errors))


@jax.jit
def train_step(
    params: optax.Params, state: optax.OptState
) -> tuple[optax.Params, optax.OptState]:
  grads = jax.grad(loss_fn)(params)
  updates, state = optimizer.update(grads, state, params)
  return optax.apply_updates(params, updates), state


initial_loss = loss_fn(train_params)
for _ in range(100):
  train_params, opt_state = train_step(train_params, opt_state)

final_loss = loss_fn(train_params)
assert bool(final_loss < initial_loss * 1e-6)
print(f'Loss: {float(initial_loss):.3f} -> {float(final_loss):.3e}')
```

The loss approaches zero and the parameters approach `target`. The optimizer
factory and `init` run outside `train_step`; JIT traces the closed-over `update`
function while the parameter and state arrays are passed as dynamic arguments.
Only array operations occur inside `update_fn`, so no Python branch depends on
the values of traced arrays.


## 4. Check the custom implementation

Compare several update steps with `optax.trace`, including a zero gradient
after a nonzero one. Momentum should decay on that step rather than disappear.
Also check that zero gradients from a fresh state give exactly zero updates.
The same checks should work with eager and JIT-compiled `update` functions.

```python
for custom_update in (momentum.update, jax.jit(momentum.update)):
  custom_state = momentum.init(params)
  reference = optax.trace(decay=decay)
  reference_state = reference.init(params)

  zero_grads = jax.tree.map(jnp.zeros_like, params)
  zero_updates, _ = custom_update(zero_grads, custom_state)
  assert optax.tree.allclose(zero_updates, zero_grads, rtol=0.0, atol=0.0)

  for multiplier in (1.0, 0.0, -2.0, 0.5):
    gradients = optax.tree.scale(multiplier, grads)
    actual, custom_state = custom_update(gradients, custom_state)
    expected, reference_state = reference.update(gradients, reference_state)
    assert optax.tree.allclose(actual, expected, rtol=1e-6, atol=1e-6)

print('Custom momentum agrees with optax.trace in eager and JIT modes.')
```

## Exercise: use the current parameters

Write a stateless transformation that adds $\lambda w_t$ to its input updates:

$$u_t = g_t + \lambda w_t.$$

This requires the current parameters. Raise a `ValueError` when `params` is
missing, and use `jax.tree.map` to combine matching parameter and update leaves.
Try implementing it before running the solution below.

```python
def add_parameter_decay(
    weight_decay: jax.typing.ArrayLike,
) -> optax.GradientTransformation:
  """Add a multiple of the current parameters to the updates.

  Args:
    weight_decay: Nonnegative scalar coefficient multiplying the parameters.

  Returns:
    A stateless transformation that requires params in its update function.

  Raises:
    ValueError: If weight_decay is a negative Python scalar, or params are
      missing at update.
  """
  if isinstance(weight_decay, (int, float)) and not weight_decay >= 0.0:
    raise ValueError('weight_decay must be nonnegative.')

  def init_fn(params: optax.Params) -> optax.EmptyState:
    del params
    return optax.EmptyState()

  def update_fn(
      updates: optax.Updates,
      state: optax.EmptyState,
      params: Optional[optax.Params] = None,
  ) -> tuple[optax.Updates, optax.EmptyState]:
    if params is None:
      raise ValueError('This transformation requires the current params.')
    updates = jax.tree.map(lambda g, p: g + weight_decay * p, updates, params)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)
```

```python
decay_transform = add_parameter_decay(0.1)
decay_state = decay_transform.init(params)
reference_decay = optax.add_decayed_weights(0.1)
reference_state = reference_decay.init(params)
expected_updates, _ = reference_decay.update(grads, reference_state, params)

for decay_update in (decay_transform.update, jax.jit(decay_transform.update)):
  decayed_updates, _ = decay_update(grads, decay_state, params)
  assert optax.tree.allclose(
      decayed_updates, expected_updates, rtol=1e-6, atol=1e-6
  )

  try:
    decay_update(grads, decay_state)
  except ValueError:
    pass
  else:
    raise AssertionError('update must reject a missing params argument.')

print(decayed_updates)
```

Optax provides this operation as [`optax.add_decayed_weights`](https://optax.readthedocs.io/en/latest/api/generated/optax.add_decayed_weights.html#optax.add_decayed_weights).
Order in a chain matters: placing it **before** momentum adds the gradient of an
$L_2$ penalty to the momentum accumulator; placing it **after** momentum and
before learning rate scaling gives decoupled weight decay.

To extend this example, replace the fixed learning rate with
`optax.exponential_decay` and inspect how the chained state evolves. See
[`optax.GradientTransformationExtraArgs`](https://optax.readthedocs.io/en/latest/api/generated/optax.GradientTransformationExtraArgs.html#optax.GradientTransformationExtraArgs) for transformations that need
additional inputs such as the current loss, and the
[transformation API](https://optax.readthedocs.io/en/latest/api/transformations.html)
for more building blocks.


## Check compatibility with hyperparameter injection

`optax.inject_hyperparams` lets hyperparameters be supplied as schedules or
changed through optimizer state. It calls the factory again inside `init`
and `update`, passing hyperparameters as arrays. Check both custom factories
with JIT-compiled initialization and updates; their results should still
match the built-in transformations.

```python
for factory, hyperparams, reference in (
    (scale_by_momentum, {'decay': decay}, optax.trace(decay=decay)),
    (
        add_parameter_decay,
        {'weight_decay': 0.1},
        optax.add_decayed_weights(0.1),
    ),
):
  injected = optax.inject_hyperparams(factory)(**hyperparams)
  injected_state = jax.jit(injected.init)(params)
  injected_update = jax.jit(injected.update)
  reference_state = reference.init(params)

  for multiplier in (1.0, 0.0, -2.0, 0.5):
    gradients = optax.tree.scale(multiplier, grads)
    actual, injected_state = injected_update(gradients, injected_state, params)
    expected, reference_state = reference.update(
        gradients, reference_state, params
    )
    assert optax.tree.allclose(actual, expected, rtol=1e-6, atol=1e-6)

print('Both custom factories support JIT with inject_hyperparams.')
```
