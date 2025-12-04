__author__ = "Joshua Windle"
__version__ = "0.1"

class Layer:
    def __init__(self):
        self.built = False

    def build(self, input_shape):
        self.built = True

    def feedforward(self, x):
        raise NotImplementedError

    def backprop(self, delta):
        raise NotImplementedError

    def update(self):
        pass
