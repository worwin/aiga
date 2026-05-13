import numpy as np


class Adam:
    """Adaptive Moment Estimation optimizer.

    Adam combines momentum with adaptive learning-rate scaling. It stores
    a moving average of gradients and a moving average of squared gradients.

    Formula:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        p_t = p_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)

    Args:
        lr: Learning rate.
        beta1: Decay rate for the first moment estimate.
        beta2: Decay rate for the second moment estimate.
        epsilon: Small value used for numerical stability.

    Attributes:
        lr: Learning rate.
        beta1: First moment decay rate.
        beta2: Second moment decay rate.
        epsilon: Numerical stability constant.
        m: First moment estimates.
        v: Second moment estimates.
        t: Update step counter.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Initialize the Adam optimizer.

        Args:
            lr: Learning rate.
            beta1: First moment decay rate.
            beta2: Second moment decay rate.
            epsilon: Numerical stability constant.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def _initialize_state(self, params):
        """Initialize moment arrays.

        Args:
            params: List of parameter arrays.
        """
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def update(self, params, grads):
        """Update parameters using Adam.

        Args:
            params: List of parameter arrays.
            grads: List of gradient arrays.
        """
        if self.m is None or self.v is None:
            self._initialize_state(params)

        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
