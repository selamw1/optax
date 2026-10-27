# Linear Neuron Boosting on MNIST

This notebook depends on the following packages:

- `equinox`
- `torch` (only for the DataLoader, can be CPU-only)
- `torchvision`
- `xdg-base-dirs`

```python
from functools import partial
import itertools
from typing import Any, Tuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import optax.contrib
import torch
import torchvision.datasets as tvdatasets
import torchvision.transforms as tvtransforms
import xdg_base_dirs

JaxInitializer: TypeAlias = jax.nn.initializers.Initializer
PyTree: TypeAlias = Any
```

## Equinox Utilities

```python
def is_neuron(node: PyTree) -> bool:
    """Returns True if the node is an Equinox module that is a linear neuron."""
    return isinstance(node, eqx.nn.Linear) or isinstance(node, eqx.nn.Conv)


def initialize(
    model: eqx.Module,
    weight_init: JaxInitializer,
    bias_init: JaxInitializer = jax.nn.initializers.zeros,
    *,
    key: jax.Array,
) -> eqx.Module:
    """Applies the initializers on the `weights` and `bias` fields for any Neuron in model.

    Args:
      model: the Module to initialize
      weight_init: the [initializer]
        (https://jax.readthedocs.io/en/latest/jax.nn.initializers.html) to use for the weights
      bias_init: the initializer for the biases, if present.
      key: the random key to split.
    """

    def _init_node(node: PyTree, key: jax.Array) -> PyTree:
        if is_neuron(node):
            weight_key, bias_key = jax.random.split(key, 2)
            weight = weight_init(weight_key, node.weight.shape, node.weight.dtype)
            node = eqx.tree_at(lambda n: n.weight, node, weight)
            if getattr(node, "bias", None) is not None:
                bias = bias_init(bias_key, node.bias.shape, node.bias.dtype)
                node = eqx.tree_at(lambda n: n.bias, node, bias)
            return node
        else:
            return node

    leaves, treedef = jtu.tree_flatten(model, is_leaf=is_neuron)
    keys = jax.random.split(key, len(leaves))
    leaves = itertools.starmap(_init_node, zip(leaves, keys, strict=True))
    return jtu.tree_unflatten(treedef, leaves)


def wrap_neurons(model: eqx.nn.MLP) -> eqx.Module:
    """Wraps model with Module whose call operator returns the tuple [output, x_neurons].

    Where x_neurons: list[Array] are the inputs to each Neuron in model.
    """
    leaves, treedef = jtu.tree_flatten(model, is_leaf=is_neuron)

    # A buffer containing the inputs to each neuron. Follows the example from:
    # https://github.com/patrick-kidger/equinox/issues/186#issuecomment-1233606690
    x_neurons = [None] * sum(map(is_neuron, leaves))

    # Saves the input to the respective location in x_neurons.
    class Neuron(eqx.Module):
        f: eqx.Module
        idx: int

        def __call__(self, x: jax.Array, *args: Any, **kwargs: Any) -> jax.Array:
            x_neurons[self.idx] = x
            return self.f(x, *args, **kwargs)

    # Wrap around the model and return each neuron's input feature, in tree traversal order.
    class Wrapper(eqx.Module):
        F: eqx.Module

        def __call__(self, *args: Any, **kwargs: Any) -> Tuple[Any, list[jax.Array]]:
            return self.F(*args, **kwargs), x_neurons

    # Replace all eqx.Modules that are linear neurons.
    idx = itertools.count()
    leaves = [Neuron(leaf, next(idx)) if is_neuron(leaf) else leaf for leaf in leaves]
    return Wrapper(jtu.tree_unflatten(treedef, leaves))
```

## Configuration

```python
SEED = 0
DATASETS_ROOT = xdg_base_dirs.xdg_data_home() / "lnb" / "mnist"
X_INVERTED = False
BATCH_SIZE = 1000

MLP_KWARGS = {
    "in_size": 28 * 28,
    "out_size": len(tvdatasets.MNIST.classes),
    "width_size": 800,
    "depth": 1,
}
WEIGHT_INIT = jax.nn.initializers.glorot_normal(in_axis=1, out_axis=0)

LNB_KWARGS = {
    "cg_ridge": 1.0,
    "is_neuron": is_neuron,
}
STEP_SIZE = 0.5
CLIP_NORM = 1.0
```

## Load MNIST

```python
def np_collate(batch):
    return jax.tree_util.tree_map(np.asarray, torch.utils.data.default_collate(batch))


def load_dataset(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), collate_fn=np_collate)
    images, labels = next(iter(loader))
    del loader
    return jnp.array(images), jnp.array(labels)


transforms = [tvtransforms.ToTensor(), tvtransforms.Lambda(torch.flatten)]
if X_INVERTED:
    transforms.append(tvtransforms.Lambda(lambda x: 1.0 - x))
transform = tvtransforms.Compose(transforms)
mnist_train = tvdatasets.MNIST(DATASETS_ROOT, download=True, train=True, transform=transform)
mnist_test = tvdatasets.MNIST(DATASETS_ROOT, download=True, train=False, transform=transform)
train_images, train_labels = load_dataset(mnist_train)
test_images, test_labels = load_dataset(mnist_test)

assert len(mnist_train) % BATCH_SIZE == 0, "Keep evenly divisible"
```

## JIT

```python
@partial(jax.jit, static_argnums=(1, 5))
def make_step(params, static, opt_state, xs, ys, opt_update):
    def loss_xs(params):
        model = eqx.combine(params, static)
        logitses, xs_neurons = jax.vmap(model)(xs)
        losses = optax.softmax_cross_entropy_with_integer_labels(logitses, ys)
        return losses.mean(), xs_neurons

    (loss, xs_neurons), grad = jax.value_and_grad(loss_xs, has_aux=True)(params)
    updates, opt_state = opt_update(grad, opt_state, params, xs_neurons=xs_neurons)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


@partial(jax.jit, static_argnums=1)
def compute_accuracy(params, static, xs, ys):
    model = eqx.combine(params, static)
    predictions = jax.vmap(lambda x: jnp.argmax(model(x)[0]))(xs)
    return jnp.mean(predictions == ys)
```

## Training loop

```python
model = eqx.nn.MLP(**MLP_KWARGS, activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
model = wrap_neurons(model)
key = jax.random.PRNGKey(SEED)
params, static = eqx.partition(model, eqx.is_inexact_array)
params = initialize(params, weight_init=WEIGHT_INIT, key=key)

opt = optax.chain(
    optax.clip_by_global_norm(CLIP_NORM),
    optax.contrib.lnb(**LNB_KWARGS),
    optax.scale_by_learning_rate(STEP_SIZE),
)
opt_state = opt.init(params)

data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=np_collate,
    generator=torch.Generator().manual_seed(SEED),
)
num_batches = len(data_loader)

t = 0
for epoch in range(200):
    for i, (xs, ys) in enumerate(data_loader):
        xs, ys = jnp.array(xs), jnp.array(ys)
        params, opt_state, loss = make_step(params, static, opt_state, xs, ys, opt.update)

        if t % 5 == 0:
            loss = loss.item()
            print(f"[{epoch}, {i:02d}/{num_batches}, {t:04d}] loss: {loss:0.5f}")
        t += 1

    train_acc = compute_accuracy(params, static, train_images, train_labels).item()
    test_acc = compute_accuracy(params, static, test_images, test_labels).item()
    print(f"[{epoch}, -----, {t:04d}] Train Acc: {train_acc:0.4f}    Test Acc: {test_acc:0.4f}")
```
