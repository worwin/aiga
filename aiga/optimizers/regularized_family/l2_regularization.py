import numpy as np


class L2Regularization:
    """L2 regularization penalty.

    L2 regularization penalizes large parameter values by adding a
    squared-parameter penalty to the objective function.

    Formula:
        R(W) = lambda * sum(W^2)

    Derivative:
        dR/dW = 2 * lambda * W

    Args:
        lambda_: Regularization strength.

    Attributes:
        lambda_: Regularization strength.
    """

    def __init__(self, lambda_=0.01):
        """Initialize L2 regularization.

        Args:
            lambda_: Regularization strength.
        """
        self.lambda_ = lambda_

    def penalty(self, params):
        """Compute the L2 penalty.

        Args:
            params: List of parameter arrays.

        Returns:
            Regularization penalty.
        """
        return self.lambda_ * sum(np.sum(p ** 2) for p in params)

    def gradient(self, params):
        """Compute the L2 regularization gradients.

        Args:
            params: List of parameter arrays.

        Returns:
            List of regularization gradients.
        """
        return [2.0 * self.lambda_ * p for p in params]
