# Aggregating and processing gradients

⚠️ **Warning**: *This is an experimental feature subject to code changes.*

Optax implements GradientTransformation that operates on average of gradients computed as the gradient of an average loss on a mini-batch. The optimizers can be seen as
$$
w_{\text{next}} = w - t_k \circ \ldots \circ t_1 \circ \text{avg} (grads)
$$
where grads here are a collection of gradients and avg is implemented implicitly by having computed the gradient on a mini-batch loss average.

The class {py:func}:`optax.experimental.aggregate.Aggregators` and the function {py:func}:`optax.experimental.aggregate.process` extend this paradigm to allow for optimizers of the form
$$
w_{\text{next}} = w -
t_k \circ \ldots \circ t_1 \circ
\text{agg} \circ  
s_j \circ \ldots \circ s_1 (grads)
$$
where the transformations $t_i$ and $s_i$ preserve the shape of their inputs (as usual gradient transformations) and agg (stands for aggregator) can for example average its inputs (so change the shape of its inputs). This paradigm enables for example a simple implementation of differential privacy setups (where individual gradients are clipped before being averaged).

In this notebook, we explain how this paradigm is implemented and present several instanciations.





```python
# @title Imports
import functools
from typing import Iterator, NamedTuple, Tuple
import jax
import jax.numpy as jnp
import jax.random as jrd

import optax
from optax import tree
from optax._src import base
from optax._src import transform
from optax._src import utils
from optax.transforms import _clipping
from optax.transforms import _combining

from optax.experimental import aggregate
```

<!-- #region -->
# Aggregators class

We introduce a new class to isolate gradient transformations that operates on per-element gradients. Its signature mimics GradientTransformationExtraArgs with `init` and `update` functions.

Optax base GradientTransformation expect input and output updates to be of the same shape as the parameters. The aggregators are supposed to take as inputs per-example gradients of shape `[*batch_shape, *params_shape]` and return an update direction of shape `[*params_shape]`. Note that we are not enforcing shape constraints during definition of instances of this class (nor we do for
GradientTransformation). This class serves as a guide to know that parts of the
rest of the optimization pipeline may need to change.


```python
PerElementUpdates = chex.ArrayTree
AggregatedUpdates = chex.ArrayTree


class AggregatorUpdateFn(Protocol):
  """Update function for aggregators."""

  def __call__(
      self,
      per_elt_updates: PerElementUpdates,
      state: base.OptState,
      params: base.Params | None = None,
      **extra_args: Any,
  ) -> tuple[AggregatedUpdates, base.OptState]:
    """Transforms per-element updates into aggregated update direction."""


class Aggregator(base.GradientTransformationExtraArgs):
  """A pair of pure functions that implement stateful aggregation of gradients.

  Attributes:
    init: Initialization function that takes params and returns state.
    update: Update function that takes per-example gradients, state and params
      (optionally) and returns updates and updated state.
  """

  init: base.TransformInitFn
  update: AggregatorUpdateFn
```

The main benefit is to let the user know what type of gradient oracle they should get.

The usual training pipeline takes the form
```python
grads = jax.grad(loss)(params, batch)
updates, opt_state = opt.update(grads, opt_state)
```
To accomodate for aggregators a simple class check suffices. Namely, one can replace the line above by
```python
if isinstance(opt, aggregators.Aggregator):
  grads = jax.vmap(jax.grad(loss), in_axes=(None, 0))(params, batch)
else:
  grads = jax.grad(loss)(params, batch)
updates, opt_state = opt.update(grads, opt_state)

```

Let's see a first basic instance with the basic `average_per_element_udpates` (defined in the `aggregate` module).

```python
def average_per_element_udpates(
    per_elt_axis: int | list[int] = 0
  ) -> aggregate.Aggregator:
  """Average per-element updates."""

  def update_fn(per_elt_updates, state, params=None):
    del params
    avg_updates = jax.tree.map(
        lambda x: jnp.mean(x, axis=per_elt_axis), per_elt_updates
    )
    return avg_updates, state

  return aggregate.Aggregator(base.init_empty_state, update_fn)
```
<!-- #endregion -->

```python
def data_iterator(
    key: jrd.PRNGKey,
    num_samples: int,
    dim: int,
    num_classes: int,
    batch_size: int,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
  """Generates a synthetic set of inputs and targets."""
  inputs_key, targets_key = jrd.split(key)
  inputs = jrd.normal(inputs_key, (num_samples, dim))
  targets = jrd.normal(targets_key, (num_samples, num_classes))

  for i in range(0, num_samples, batch_size):
    yield inputs[i : i + batch_size], targets[i : i + batch_size]


def loss_fun(params, batch):
  inputs, targets = batch
  return jnp.mean(jnp.sum((inputs.dot(params) - targets) ** 2, -1))


def basic_train(opt):
  num_samples, batch_size, dim, num_classes = 16, 4, 4, 2

  data = data_iterator(jrd.key(0), num_samples, dim, num_classes, batch_size)
  params = jrd.normal(jrd.key(1), (dim, num_classes))

  @jax.jit
  def train_step(params, state, batch):
    if isinstance(opt, aggregate.Aggregator):
      losses, grads = jax.vmap(jax.value_and_grad(loss_fun), (None, 0))(
          params, batch
      )
      loss = jnp.mean(losses)
    else:
      loss, grads = jax.value_and_grad(loss_fun)(params, batch)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)
    return params, state, loss

  state = opt.init(params)
  for i, batch in enumerate(data):
    params, state, loss = train_step(params, state, batch)
    print(f'Step: {i} | Batch loss: {loss:.2e}')
```

```python
print('Standard training')
opt = optax.sgd(learning_rate=0.01)

basic_train(opt)

print('\nWith explicit aggregation')
opt = aggregate.chain(
    aggregate.average_per_element_udpates(), optax.sgd(learning_rate=0.01)
)
basic_train(opt)
```

<!-- #region -->
## Processing gradients

Optimizers of the form
$$
w_{\text{next}} = w -
t_k \circ \ldots \circ t_1 \circ
\text{agg} \circ  
s_j \circ \ldots \circ s_1 (grads)
$$
may be defined just a chain of gradient transformations.

However,
1. we may aggregate gradients by doing simple gradient accumulation rather than computing all gradients at once,
2. we may want to do more than just average the gradients (for example we may want to access to some variance).


For this reason, we provide a `process` function.
```python
def process(
    preprocessor: base.GradientTransformation,
    aggregator: base.GradientTransformation | Aggregator,
    postprocessor: base.GradientTransformation,
    aggregator_has_aux: bool = False,
):
  """Process gradients through a sequence of transformations.

  Args:
    preprocessor: A transformation that maps per-example gradients to
      per-example updates.
    aggregator: A transformation that aggregates per-example updates into a
      single update.
    postprocessor: A transformation that maps aggregated updates to the final
      updates.
    aggregator_has_aux: Whether the aggregator returns more than just the
      average updates.

  Returns:
    A :class:`optax.GradientTransformation`.
  """

  def init_fn(params) -> tuple[base.OptState, base.OptState, base.OptState]:
    preprocess_state = preprocessor.init(params)
    aggregate_state = aggregator.init(params)
    postprocess_state = postprocessor.init(params)
    return preprocess_state, aggregate_state, postprocess_state

  def update_fn(indiv_grads, states, params=None, **extra_args):
    preprocess_state, aggregate_state, postprocess_state = states

    indiv_updates, new_preprocess_state = preprocessor.update(
        indiv_grads, preprocess_state, params, **extra_args
    )

    aggregated, new_aggregate_state = aggregator.update(
        indiv_updates, aggregate_state, params, **extra_args
    )

    if aggregator_has_aux:
      avg_updates, agg_aux = aggregated
      extra_args = extra_args | agg_aux
    else:
      avg_updates = aggregated

    ready_to_post_process = tree.get(new_aggregate_state, 'ready', True)

    updates, new_postprocess_state = jax.lax.cond(
        ready_to_post_process,
        lambda g, s, p, kw: postprocessor.update(g, s, p, **kw),
        lambda g, s, *_: (tree.zeros_like(avg_updates), s),
        avg_updates,
        postprocess_state,
        params,
        extra_args,
    )
    return updates, (
        new_preprocess_state,
        new_aggregate_state,
        new_postprocess_state,
    )

  if isinstance(aggregator, Aggregator):
    return Aggregator(init_fn, update_fn)
  else:
    return base.GradientTransformationExtraArgs(init_fn, update_fn)
```

This function lets the user define an `aggregate` transform that can aggregate gradients it receives chunk by chunk until is ready to post-process them.
It also provides the possibility to pass along more than the average updates to
the post-processing stage.

As a simple example, we can extend the basic `average_per_element_updates` to
work with micro-batches using the following tools (in the `aggregate` module).


```python
class AccumulateAvgUpdatesState(NamedTuple):
  """State for the average gradient accumulator."""

  micro_step: int
  ready: bool
  avg_grad: base.Updates


def accumulate_avg_udpates(
    num_microbatches: int,
) -> base.GradientTransformation:
  """Accumulate average gradients."""

  if num_microbatches < 1:
    raise ValueError('num_microbatches must be larger than or equal to than 0.')

  if num_microbatches == 1:
    # If there is only one microbatch, we don't need accumulation.
    # We return identity to save unnecessary state tracking.
    return base.identity()

  def init_fn(params):
    return AccumulateAvgUpdatesState(
        micro_step=0, ready=False, avg_grad=tree.zeros_like(params)
    )

  def update_fn(updates, state, params=None):
    del params
    new_micro_step = state.micro_step + 1
    new_avg_grad = jax.tree.map(
        lambda u, a: a + (u - a) / new_micro_step,
        updates,
        state.avg_grad,
    )
    ready_state = AccumulateAvgUpdatesState(
        micro_step=0, ready=True, avg_grad=tree.zeros_like(new_avg_grad)
    )
    not_ready_state = AccumulateAvgUpdatesState(
        micro_step=new_micro_step, ready=False, avg_grad=new_avg_grad
    )
    updates, new_state = tree.where(
        new_micro_step == num_microbatches,
        (new_avg_grad, ready_state),
        (tree.zeros_like(new_avg_grad), not_ready_state),
    )
    return updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def average_incrementally_updates(
    per_elt_axis: int | list[int] | None, num_microbatches: int
) -> Aggregator | base.GradientTransformation:
  """Average and accumulate per-element updates."""
  if per_elt_axis is None:
    return accumulate_avg_udpates(num_microbatches)
  else:
    return chain(
        average_per_element_udpates(per_elt_axis),
        accumulate_avg_udpates(num_microbatches),
    )
```

We can revise our basic example with this.
<!-- #endregion -->

```python
def train(
    opt,
    num_microbatches: int = 1,
    num_samples: int = 16,
    batch_size: int = 4,
    dim: int = 4,
    num_classes: int = 2,
):

  data_iter = lambda: data_iterator(
      jrd.key(0), num_samples, dim, num_classes, batch_size // num_microbatches
  )
  full_data = [jnp.concatenate(a, axis=0) for a in zip(*data_iter())]
  params = jrd.normal(jrd.key(1), (dim, num_classes))

  @jax.jit
  def train_step(params, state, batch):
    if isinstance(opt, aggregate.Aggregator):
      losses, grads = jax.vmap(jax.value_and_grad(loss_fun), (None, 0))(
          params, batch
      )
      loss = jnp.mean(losses)
    else:
      loss, grads = jax.value_and_grad(loss_fun)(params, batch)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)
    return params, state, loss

  state = opt.init(params)
  for i, batch in enumerate(data_iter()):
    full_batch_loss = loss_fun(params, full_data)
    params, state, loss = train_step(params, state, batch)
    print(
        f'Step: {i} |'
        f'Mini-batch Loss: {loss:.2e} |'
        f'Full batch loss: {full_batch_loss:.2e}'
    )
```

```python
print('Standard training')
opt = optax.sgd(learning_rate=0.01)
train(opt)

print('\nWith accumulation')
num_microbatches = 2
# The optimizer below does not use per-example average as we use
# per_elt_axis=None. It returns a standard GradientTransform and uses
# jax.grad in the train pipeline above
opt = aggregate.process(
    base.identity(),
    aggregate.average_incrementally_updates(
        per_elt_axis=None, num_microbatches=num_microbatches
    ),
    optax.sgd(learning_rate=0.01),
)
train(opt, num_microbatches)

print('\nWith explicit aggregation and accumulation')
# This optimizer is an Aggregator and will sue the vmap grads
opt = aggregate.process(
    base.identity(),
    aggregate.average_incrementally_updates(
        per_elt_axis=0, num_microbatches=num_microbatches),
    optax.sgd(learning_rate=0.01),
)
train(opt, num_microbatches)

```

The resulting mini-batch losses between with or without accumulation
do not match since the mini-batches are not the same and we are not
accumulating losses as we are accumulating gradients.
The full losses naturally match: the full batch loss at step `i` without accumulation matches the full batch loss with accumulation at step `num_microbatches x i`.

<!-- #region -->
Note that the proposed `accumulate_avg_udpates` combined with `process` can also replace `optax.MultiSteps` as

```python
def accumulate_grads(
    opt: base.GradientTransformation,
    num_microbatches: int,
) -> base.GradientTransformation:
  """Accumulate gradients."""
  return process(
      preprocessor=base.identity(),
      aggregator=average_incrementally_updates(
          per_elt_axis=None,
          num_microbatches=num_microbatches,
      ),
      postprocessor=opt,
  )
```

In the rest of the notebook, we present how this paradigm can be applied (i) in differential privacy, (ii) to record variance of gradients per coordinate.
<!-- #endregion -->

## Differentially private SGD

Differentially private sgd is an algorithm that clips per-example gradients, then add noise to preserve privacy (see [Deep Learning with Differential Privacy (Abadi et al., 2016)](https://arxiv.org/abs/1607.00133) for more details). With the help of the `process` function such an algorithm can be implemented very easily.

```python
def per_example_clip(
    l2_norm_clip: float,
    per_elt_axis: int | list[int] | None = 0,
) -> base.GradientTransformation:
  """Clip per-example gradients with their individual norm."""

  if per_elt_axis is None:
    return _clipping.clip_by_global_norm(l2_norm_clip)

  def update_fn(per_elt_grads, state, params=None):
    del params
    clip = functools.partial(
        optax.projections.projection_l2_ball, scale=l2_norm_clip
    )
    clipped_updates = jax.vmap(clip, in_axes=per_elt_axis)(per_elt_grads)
    return clipped_updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def differentially_private_aggregate(
    l2_norm_clip: float,
    noise_multiplier: float,
    key: jax.Array | int,
    per_elt_axis: int | list[int] | None = 0,
    num_microbatches: int = 1,
) -> base.GradientTransformation | aggregate.Aggregator:
  """Processes gradients based on the DPSGD algorithm."""
  noise_std = l2_norm_clip * noise_multiplier

  return aggregate.process(
      preprocessor=per_example_clip(l2_norm_clip, per_elt_axis),
      aggregator=aggregate.average_incrementally_updates(
          per_elt_axis=per_elt_axis,
          num_microbatches=num_microbatches,
      ),
      postprocessor=transform.add_noise(
          eta=noise_std,
          gamma=1.0,
          key=key,
      ),
  )
```

We can instantiate it the pipeline defined before.

```python
print('Without DP')
train(optax.sgd(learning_rate=0.01))

print('\nWith DP')
opt = aggregate.chain(
    differentially_private_aggregate(
        l2_norm_clip=1.0, noise_multiplier=1.0, key=jrd.key(2)
    ),
    optax.sgd(learning_rate=0.01),
)
train(opt)
```

## Micro-Adam

We can instantiate a variant of Adam that uses average of square rather than square of average gradients (see [Batch size invariant Adam (Wang & Aitchison, 2024)](https://arxiv.org/abs/2402.18824)).

This instance requires us to aggregate both average gradient and average of square gradients. So it illustrates the need of a `has_aux` argument for the `process` function.


```python
def scale_by_micro_adam(
    b1: float,
    b2: float,
    eps: float,
) -> base.GradientTransformationExtraArgs:
  """Micro-Adam optimizer."""

  def init_fn(params):
    return transform.scale_by_adam(b1=b1, b2=b2, eps=eps).init(params)

  def update_fn(updates, state, params=None, *, avg_sq_updates, **extra_args):
    del params, extra_args
    mu = tree.update_moment(updates, state.mu, b1, 1)
    nu = tree.update_moment(avg_sq_updates, state.nu, b2, 1)
    count_inc = utils.safe_int32_increment(state.count)

    mu_hat = tree.bias_correction(mu, b1, count_inc)
    nu_hat = tree.bias_correction(nu, b2, count_inc)
    updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
    new_state = transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)
    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def get_avg_and_avg_sq_updates(
    per_elt_axis: int | list[int] | None = 0,
    num_microbatches: int = 1,
) -> base.GradientTransformationExtraArgs:
  """Collect average and average of squares of gradients."""

  def incremental_update_fn(updates, state, params=None):
    # With this update function, we use the accumulation below to progressively
    # compute the average and average of squares of updates.
    del params
    sq_updates = jax.tree.map(jnp.square, updates)
    return (updates, {'avg_sq_updates': sq_updates}), state

  def per_elt_update_fn(per_elt_updates, state, params=None):
    # With this update function, we consider `per_elt_updates` to be per-element
    # updates that we want to average on. The accumulator below may even let us
    # get reach larger batches.
    del params
    avg_updates = jax.tree.map(
        lambda x: jnp.mean(x, axis=per_elt_axis), per_elt_updates
    )
    avg_sq_updates = jax.tree.map(
        lambda x: jnp.mean(jnp.square(x), axis=per_elt_axis), per_elt_updates
    )
    return (avg_updates, {'avg_sq_updates': avg_sq_updates}), state

  if per_elt_axis is None:
    opt = aggregate.chain(
        base.GradientTransformation(
            base.init_empty_state, incremental_update_fn
        ),
        aggregate.accumulate_avg_udpates(num_microbatches),
    )
  else:
    opt = aggregate.chain(
        aggregate.Aggregator(base.init_empty_state, per_elt_update_fn),
        aggregate.accumulate_avg_udpates(num_microbatches),
    )
  return opt


def micro_adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    per_elt_axis: int | list[int] | None = 0,
    num_microbatches: int = 1,
) -> base.GradientTransformation:
  """Micro-Adam optimizer."""
  return aggregate.process(
      preprocessor=base.identity(),
      aggregator=get_avg_and_avg_sq_updates(per_elt_axis, num_microbatches),
      postprocessor=aggregate.chain(
          scale_by_micro_adam(b1, b2, eps),
          transform.scale_by_learning_rate(learning_rate),
      ),
      aggregator_has_aux=True,
  )

```

```python
print('Classical Adam')
opt = optax.adam(learning_rate=0.01)
train(opt)

print('Micro-Adam')
opt = micro_adam(learning_rate=0.01)
train(opt)

```

## Adam with variance computations

Finally, we present how to track per element gradient variance during optimization. The variance is accumulated in a numerically stable manner with Welford's algorithm.

```python
def get_batch_size_from_per_elt_updates(
    per_elt_updates: base.Updates, per_elt_axis: int | list[int]
) -> int:
  """Get batch size from per-element updates."""

  def get_batch_size(u):
    if isinstance(per_elt_axis, int):
      return u.shape[per_elt_axis]
    else:
      return functools.reduce(
          lambda a, b: a * b, [u.shape[i] for i in per_elt_axis]
      )

  batch_sizes = jax.tree.map(get_batch_size, per_elt_updates)
  batch_sizes = jax.tree.leaves(batch_sizes)
  if not all(b == batch_sizes[0] for b in batch_sizes):
    raise ValueError(
        f'Per-element updates must have the same batch size. Got: {batch_sizes}'
    )
  return batch_sizes[0]


class PerElementMeanAndSumSqDiffGradsState(NamedTuple):
  """State for the per-element mean and variance accumulator."""

  micro_step: int
  ready: bool
  mean_grads: base.Updates
  sum_sq_diff_grads: base.Updates


def get_per_element_mean_and_sum_sq_diff_grads(
    per_elt_axis: int | list[int] = 0,
    num_microbatches: int = 1,
) -> aggregate.Aggregator:
  """Collect per-element mean and variance gradient metrics."""

  if per_elt_axis is None:
    raise NotImplementedError(
        'Per-element mean and sum square diff need a per_elt_axis.'
    )

  def compute_avg_and_sum_sq_diff(
      per_elt_udpates: base.Updates,
      state: base.OptState,
      params: base.Params | None,
  ) -> tuple[base.Updates, base.Updates]:
    del params
    batch_size = get_batch_size_from_per_elt_updates(
        per_elt_udpates, per_elt_axis
    )
    mean_grads = jax.tree.map(
        lambda x: jnp.mean(x, axis=per_elt_axis), per_elt_udpates
    )
    sum_sq_diff_grads = jax.tree.map(
        lambda x: jnp.sum(jnp.square(x), axis=per_elt_axis), per_elt_udpates
    )
    return (
        mean_grads,
        {'sum_sq_diff_grads': sum_sq_diff_grads, 'sample_size': batch_size},
    ), state

  if num_microbatches == 1:
    return aggregate.Aggregator(
        base.init_empty_state, compute_avg_and_sum_sq_diff
    )

  def init_fn(params):
    return PerElementMeanAndSumSqDiffGradsState(
        micro_step=0,
        ready=False,
        mean_grads=tree.zeros_like(params),
        sum_sq_diff_grads=tree.zeros_like(params),
    )

  def update_fn(per_elt_udpates, state, params=None):
    del params
    batch_size = get_batch_size_from_per_elt_updates(
        per_elt_udpates, per_elt_axis
    )
    new_micro_step = state.micro_step + 1

    # Compute batch averages.
    batch_mean_grads = jax.tree.map(
        lambda x: jnp.mean(x, axis=per_elt_axis, keepdims=True), per_elt_udpates
    )
    batch_sum_sq_diff_grads = jax.tree.map(
        lambda x, a: jnp.sum(jnp.square(x - a), axis=per_elt_axis),
        per_elt_udpates,
        batch_mean_grads,
    )
    batch_mean_grads = jax.tree.map(
        lambda x: x.squeeze(axis=per_elt_axis), batch_mean_grads
    )

    # Update accumulated averages.
    delta = jax.tree.map(lambda u, a: u - a, batch_mean_grads, state.mean_grads)
    new_mean_grads = jax.tree.map(
        lambda a, d: a + d / new_micro_step,
        state.mean_grads,
        delta,
    )
    size_factor = state.micro_step * batch_size / new_micro_step
    new_sum_sq_diff_grads = jax.tree.map(
        lambda a, s, d: a + s + d**2 * size_factor,
        state.sum_sq_diff_grads,
        batch_sum_sq_diff_grads,
        delta,
    )
    maybe_outputs = (
        new_mean_grads,
        {
            'sum_sq_diff_grads': new_sum_sq_diff_grads,
            'sample_size': batch_size * new_micro_step,
        },
    )

    # Output or not the accumulated averages.
    ready_state = PerElementMeanAndSumSqDiffGradsState(
        micro_step=0,
        ready=True,
        mean_grads=tree.zeros_like(new_mean_grads),
        sum_sq_diff_grads=tree.zeros_like(new_sum_sq_diff_grads),
    )
    not_ready_state = PerElementMeanAndSumSqDiffGradsState(
        micro_step=new_micro_step,
        ready=False,
        mean_grads=new_mean_grads,
        sum_sq_diff_grads=new_sum_sq_diff_grads,
    )
    updates, new_state = tree.where(
        new_micro_step == num_microbatches,
        (maybe_outputs, ready_state),
        (tree.zeros_like(maybe_outputs), not_ready_state),
    )
    return updates, new_state

  return aggregate.Aggregator(init_fn, update_fn)


class PerElementMeanAndVarianceEMAState(NamedTuple):
  """State for the per-element mean and variance accumulator."""

  count: jax.Array
  ema_decay: jax.Array
  mean_grads_ema: base.Updates
  variance_grads_ema: base.Updates


def track_per_element_mean_and_variance_with_ema(
    ema_decay: float = 0.9,
) -> base.GradientTransformation:
  """Track variance metrics with an EMA over time."""

  def init_fn(params):
    return PerElementMeanAndVarianceEMAState(
        count=jnp.zeros([], jnp.int32),
        ema_decay=jnp.asarray(ema_decay),
        mean_grads_ema=tree.zeros_like(params),
        variance_grads_ema=tree.zeros_like(params),
    )

  def update_fn(updates, state, params=None, *, sum_sq_diff_grads, sample_size):
    del params
    mean_grads_ema = jax.tree.map(
        lambda x, y: (1.0 - ema_decay) * x + ema_decay * y,
        updates,
        state.mean_grads_ema,
    )
    variance_step = tree.scale(1 / (sample_size - 1), sum_sq_diff_grads)
    variance_grads_ema = jax.tree.map(
        lambda x, y: (1.0 - ema_decay) * x + ema_decay * y,
        variance_step,
        state.variance_grads_ema,
    )
    new_count = utils.safe_int32_increment(state.count)
    new_state = state._replace(
        count=new_count,
        mean_grads_ema=mean_grads_ema,
        variance_grads_ema=variance_grads_ema,
    )
    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def get_unbiased_mean_and_variance_ema(
    state: base.OptState,
) -> tuple[base.Updates, base.Updates]:
  """Track unbiased mean and variance with an EMA over time."""
  per_elt_mean_and_variance_ema_state = tree.get(
      state, 'PerElementMeanAndVarianceEMAState', None
  )
  if per_elt_mean_and_variance_ema_state is None:
    raise ValueError(
        'State must have PerElementMeanAndVarianceEMAState to compute unbiased'
        ' mean and variance EMA.'
    )
  count = per_elt_mean_and_variance_ema_state.count
  ema_decay = per_elt_mean_and_variance_ema_state.ema_decay
  mean_grads_ema = per_elt_mean_and_variance_ema_state.mean_grads_ema
  variance_grads_ema = per_elt_mean_and_variance_ema_state.variance_grads_ema
  unbiased_mean_grads_ema = jax.tree.map(
      lambda x: x / (1 - ema_decay**count), mean_grads_ema
  )
  unbiased_variance_grads_ema = jax.tree.map(
      lambda x: x / (1 - ema_decay**count), variance_grads_ema
  )
  return unbiased_mean_grads_ema, unbiased_variance_grads_ema


def add_mean_variance_to_opt(
    opt: base.GradientTransformation,
    ema_decay: float = 0.9,
    per_elt_axis: int | list[int] | None = 0,
    num_microbatches: int = 1,
):
  """Add mean and variance to an optimizer."""
  return aggregate.process(
      preprocessor=base.identity(),
      aggregator=get_per_element_mean_and_sum_sq_diff_grads(
          per_elt_axis, num_microbatches
      ),
      postprocessor=aggregate.chain(
          track_per_element_mean_and_variance_with_ema(ema_decay),
          opt,
      ),
      aggregator_has_aux=True,
  )
```

```python
def train_and_track(
    opt,
    num_microbatches: int = 1,
    num_samples: int = 16,
    batch_size: int = 4,
    dim: int = 4,
    num_classes: int = 2,
):

  data_iter = lambda: data_iterator(
      jrd.key(0), num_samples, dim, num_classes, batch_size // num_microbatches
  )
  full_data = [jnp.concatenate(a, axis=0) for a in zip(*data_iter())]
  params = jrd.normal(jrd.key(1), (dim, num_classes))

  opt = add_mean_variance_to_opt(opt)

  @jax.jit
  def train_step(params, state, batch):
    if isinstance(opt, aggregate.Aggregator):
      losses, grads = jax.vmap(jax.value_and_grad(loss_fun), (None, 0))(
          params, batch
      )
      loss = jnp.mean(losses)
    else:
      loss, grads = jax.value_and_grad(loss_fun)(params, batch)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)
    return params, state, loss

  state = opt.init(params)
  for i, batch in enumerate(data_iter()):
    full_batch_loss = loss_fun(params, full_data)
    params, state, loss = train_step(params, state, batch)
    mean, var = get_unbiased_mean_and_variance_ema(state)
    print(
        f'Step: {i} |'
        f'Mini-batch Loss: {loss:.2e} |'
        f'Full batch loss: {full_batch_loss:.2e}\n'
        f'Mean EMA:\n {mean}\n'
        f'Variance EMA:\n {var}:'
    )
```

```python
train_and_track(optax.adam(learning_rate=1e-1))
```

# Sharp edges

Currently, the axis along which both the aggregator is done is not accessible where the train step is defined. Ideally GradientTransforms would be DataClasses so Aggregators could store meta-fields like the axis along which a vmap can be done.
