import numpy as np

from .layer import Layer


class Dropout(Layer):
    """Dropout layer using inverted scaling during training.

    During training:
        out = x * mask / (1 - p)
    During evaluation:
        out = x
    """

    def __init__(self, p=0.5, seed=None):
        super().__init__()

        if p < 0.0 or p >= 1.0:
            raise ValueError("Dropout probability p must satisfy 0.0 <= p < 1.0.")

        self.p = p
        self.seed = seed
        self.training = True
        self.mask = None
        self._rng = np.random.default_rng(seed)

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self.mask = np.ones_like(x, dtype=float)
            return x

        keep_prob = 1.0 - self.p
        self.mask = (self._rng.random(x.shape) < keep_prob).astype(float)
        return x * self.mask / keep_prob

    def backprop(self, delta):
        if self.mask is None:
            raise ValueError("Forward must be called before backprop.")

        if not self.training or self.p == 0.0:
            return delta

        keep_prob = 1.0 - self.p
        return delta * self.mask / keep_prob
