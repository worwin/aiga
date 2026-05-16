import numpy as np


class AdamW:
    """AdamW optimizer with decoupled weight decay.

    AdamW follows Adam's adaptive moment updates and applies weight decay as a
    separate parameter shrinkage step (decoupled from the gradient update).

    Formula:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        p_t = p_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
        p_t = p_t - lr * weight_decay * p_t
    """

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def _initialize_state(self, params):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def update(self, params, grads):
        if self.m is None or self.v is None:
            self._initialize_state(params)

        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            p -= self.lr * self.weight_decay * p
