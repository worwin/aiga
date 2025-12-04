__author__ = "Joshua Windle"
__version__ = "0.1"

import numpy as np
from .layer import Layer


class Dense(Layer):
    def __init__(self, output_size, input_size=None):
        super().__init__()              # executes constructor of the parent Layer class
        self.output_size = output_size  # the number of output neurons        
        self.input_size = input_size    # the number of input features

        self.W = None                   # the weight matrix W (linear map)
        self.b = None                   # the bias vector (shifts affine transformation)

        self.dW = None                  # gradient of ∂C/∂W
        self.db = None                  # gradient of ∂C/∂b

        self.a = None                   # cache for backprop

        if input_size is not None:
            self.build(input_size)

    def __call__(self, a):
        return self.forward(a)

    def build(self, input_size):
        self.input_size = input_size

        # Small random init
        self.W = 0.01 * np.random.randn(input_size, self.output_size)
        self.b = np.zeros(self.output_size)

        self.built = True

    def forward(self, a):
        if not self.built:
            self.build(a.shape[-1])

        self.a = a
        return a @ self.W + self.b



    def backprop(self, delta):
        # Gradients
        self.dW = self.a.T @ delta
        self.db = np.sum(delta, axis=0)

        # Propagate error backward
        return delta @ self.W.T

    def update(self, optim):
        optim.update([self.W, self.b], [self.dW, self.db])

