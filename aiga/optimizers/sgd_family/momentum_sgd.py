import numpy as np


class MomentumSGD:
    """Stochastic gradient descent with momentum.

    MomentumSGD extends plain SGD by storing a velocity term for each
    parameter. The velocity remembers recent gradient directions and
    smooths the update path across training steps.

    Formula:
        v_t = gamma * v_{t-1} + lr * g_t
        p_t = p_{t-1} - v_t

    Args:
        lr: Learning rate used to scale the gradient.
        gamma: Momentum coefficient used to retain previous velocity.

    Attributes:
        lr: Learning rate.
        gamma: Momentum coefficient.
        velocities: Stored velocity arrays for each parameter.
    """

    def __init__(self, lr=0.01, gamma=0.9):
        """Initialize the MomentumSGD optimizer.

        Args:
            lr: Learning rate.
            gamma: Momentum coefficient.
        """
        self.lr = lr
        self.gamma = gamma
        self.velocities = None

    def _initialize_velocities(self, params):
        """Initialize velocity arrays.

        Args:
            params: List of parameter arrays.
        """
        self.velocities = [np.zeros_like(p) for p in params]

    def update(self, params, grads):
        """Update parameters using momentum SGD.

        Args:
            params: List of parameter arrays.
            grads: List of gradient arrays.
        """
        if self.velocities is None:
            self._initialize_velocities(params)

        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocities[i] = self.gamma * self.velocities[i] + self.lr * g
            p -= self.velocities[i]
