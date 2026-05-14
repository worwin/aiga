import numpy as np


class AdaGrad:
    """Adaptive Gradient optimizer.

    AdaGrad adapts the learning rate for each parameter by accumulating
    the squared gradients seen so far. Parameters with large historical
    gradients receive smaller future updates.

    Formula:
        r_t = r_{t-1} + g_t^2
        p_t = p_{t-1} - lr * g_t / (sqrt(r_t) + epsilon)

    Args:
        lr: Base learning rate.
        epsilon: Small value used for numerical stability.

    Attributes:
        lr: Base learning rate.
        epsilon: Numerical stability constant.
        accumulators: Accumulated squared gradients for each parameter.
    """

    def __init__(self, lr=0.01, epsilon=1e-8):
        """Initialize the AdaGrad optimizer.

        Args:
            lr: Base learning rate.
            epsilon: Numerical stability constant.
        """
        self.lr = lr
        self.epsilon = epsilon
        self.accumulators = None

    def _initialize_accumulators(self, params):
        """Initialize accumulated squared-gradient arrays.

        Args:
            params: List of parameter arrays.
        """
        self.accumulators = [np.zeros_like(p) for p in params]

    def update(self, params, grads):
        """Update parameters using AdaGrad.

        Args:
            params: List of parameter arrays.
            grads: List of gradient arrays.
        """
        if self.accumulators is None:
            self._initialize_accumulators(params)

        for i, (p, g) in enumerate(zip(params, grads)):
            self.accumulators[i] += g ** 2
            p -= self.lr * g / (np.sqrt(self.accumulators[i]) + self.epsilon)
