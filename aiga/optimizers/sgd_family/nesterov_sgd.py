import numpy as np


class NesterovSGD:
    """Stochastic gradient descent with Nesterov-style momentum.

    Nesterov accelerated gradient modifies classical momentum by applying
    a look-ahead style correction to the parameter update. This class uses
    the common practical NAG update form that can fit the same optimizer
    interface as SGD.

    Formula:
        v_t = gamma * v_{t-1} + lr * g_t
        p_t = p_{t-1} - ((1 + gamma) * v_t - gamma * v_{t-1})

    Args:
        lr: Learning rate used to scale the gradient.
        gamma: Momentum coefficient used to retain previous velocity.

    Attributes:
        lr: Learning rate.
        gamma: Momentum coefficient.
        velocities: Stored velocity arrays for each parameter.
    """

    def __init__(self, lr=0.01, gamma=0.9):
        """Initialize the NesterovSGD optimizer.

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
        """Update parameters using Nesterov-style momentum.

        Args:
            params: List of parameter arrays.
            grads: List of gradient arrays.
        """
        if self.velocities is None:
            self._initialize_velocities(params)

        for i, (p, g) in enumerate(zip(params, grads)):
            previous_velocity = self.velocities[i].copy()
            self.velocities[i] = self.gamma * self.velocities[i] + self.lr * g
            p -= (1.0 + self.gamma) * self.velocities[i] - self.gamma * previous_velocity
