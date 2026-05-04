__author__ = "Joshua Windle"
__version__ = "0.1"

class Layer:
    # Most layers are not trainable by default.
    # Example: ReLU has no weights or biases, so it does not need update().
    trainable = False

    def __init__(self):
        self.built = False

    def build(self, input_shape):
        self.built = True

    def forward(self, x):
        raise NotImplementedError

    def backprop(self, delta):
        raise NotImplementedError