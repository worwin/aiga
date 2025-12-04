__author__ = "Joshua W. Windle"
__version__ = "0.1"

import numpy as np


class MSE:
    def forward(self, y_hat, y):
        """
        Mean Squared Error loss:

            C = (1 / N) * sum((y_hat - y)^2)
        """
        self.y_hat = y_hat
        self.y = y
        return np.mean((y_hat - y) ** 2)

    def backprop(self, y_hat=None, y=None):
        """
        Derivative of MSE w.r.t. network output:

            dC/dy_hat = (2 / N) * (y_hat - y)
        """
        if y_hat is None:
            y_hat = self.y_hat
        if y is None:
            y = self.y

        N = y.shape[0]

        # ∂C/∂y_hat
        delta = (2 / N) * (y_hat - y)

        return delta
