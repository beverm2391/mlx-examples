# Copyright © 2023 Apple Inc.

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mnist


class BatchNorm2d(nn.Module):
    r"""Applies batch normalization on the inputs.

    Normalizes the input using the mean and variance of the batch and applies
    a scale and shift transformation. The mean and variance are computed 
    across the batch for each feature independently.

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
        momentum (float): Momentum for the running mean and variance
        affine (bool): If True, learn an affine transform to apply after the
            normalization
    """
    def __init__(
        self,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        momentum: float = 0.1
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.dims = dims
        if affine:
            self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.running_mean = mx.zeros((dims,))
        self.running_var = mx.ones((dims,))

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, momentum={self.momentum}, affine={'weight' in self}"
    
    def __call__(self, x, training=True):
        if training:
            batch_mean = mx.mean(x, axis=0)
            batch_var = mx.mean(x, axis=0)
            self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * batch_var + (1 - self.momentum) * self.running_var
            normalized_x = (x - batch_mean) * mx.rsqrt(batch_var + self.eps)
        else:
            normalized_x = (x - self.running_mean) * mx.rsqrt(self.running_var + self.eps)

        return (self.weight * normalized_x + self.bias) if "weight" in self else normalized_x

class MaxPool2D(nn.Module):
    r"""Applies a 2D max pooling over an input signal (e.g., an image).

    Args:
        kernel_size (int or tuple): Size of the pooling window
        stride (int or tuple, optional): Stride of the pooling window
        padding (int or tuple, optional): Implicit zero padding to be added on both sides
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0
    ):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def _extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"

    def __call__(self, x):
        # Call the underlying C++ implementation
        return mx.max_pool2d(x, self.kernel_size, self.stride, self.padding)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Conv2d(1, 32, 5),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 5),
            nn.ReLU(),
            BatchNorm2d(32),
            # maxpool2d
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            BatchNorm2d(64),
            # maxpool2d
            lambda x: x.reshape(-1), # flatten 
            nn.Linear(576, 10)
        ]

    def __call__(self, x):
        return nn.Sequential(*self.layers)(x)


def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


def main():
    seed = 0
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    np.random.seed(seed)

    # Load the data
    train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())

    # Load the model
    model = Model(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    for e in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
        accuracy = eval_fn(model, test_images, test_labels)
        toc = time.perf_counter()
        print(
            f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
            f" Time {toc - tic:.3f} (s)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple MLP on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main()