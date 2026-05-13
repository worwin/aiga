__author__ = "Joshua W. Windle"
__version__ = "0.1"


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        """
        params: list of parameters [W, b, ...]
        grads:  list of gradients  [dW, db, ...]
        """
        for p, g in zip(params, grads):
            p -= self.lr * g
