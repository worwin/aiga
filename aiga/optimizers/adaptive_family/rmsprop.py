import numpy as np


class RMSProp:
    """Root Mean Square Propagation optimizer.

    RMSProp is an AdaGrad-style optimizer that uses an exponentially
    decaying average of squared gradients instead of accumulating all
    squared gradients forever.

    Formula:
        r_t = beta * r_{t-1} + (1 - beta) * g_t^2
        p_t = p_{t-1} - lr * g_t / (sqrt(r_t) + epsilon)

    Args:
        lr: Base learning rate.
        beta: Decay rate for the squared-gradient moving average.
        epsilon: Small value used for numerical stability.

    Attributes:
        lr: Base learning rate.
        beta: Squared-gradient decay rate.
        epsilon: Numerical stability constant.
        averages: Moving averages of squared gradients.
    """

    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        """Initialize the RMSProp optimizer.

        Args:
            lr: Base learning rate.
            beta: Squared-gradient decay rate.
            epsilon: Numerical stability constant.
        """
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.averages = None

    def _initialize_averages(self, params):
        """Initialize squared-gradient moving averages.

        Args:
            params: List of parameter arrays.
        """
        self.averages = [np.zeros_like(p) for p in params]

    def update(self, params, grads):
        """Update parameters using RMSProp.

        Args:
            params: List of parameter arrays.
            grads: List of gradient arrays.
        """
        if self.averages is None:
            self._initialize_averages(params)

        for i, (p, g) in enumerate(zip(params, grads)):
            self.averages[i] = self.beta * self.averages[i] + (1.0 - self.beta) * (g ** 2)
            p -= self.lr * g / (np.sqrt(self.averages[i]) + self.epsilon)
