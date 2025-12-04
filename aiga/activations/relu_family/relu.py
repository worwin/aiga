import numpy as np
from aiga.layers.layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.a = None   # cached input

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        return np.maximum(0, a)

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        return (self.a > 0).astype(float) * delta

    def update(self, lr=0.01):
        pass   # no parameters to update
