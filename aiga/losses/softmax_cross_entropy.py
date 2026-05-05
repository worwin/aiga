import numpy as np


class SoftmaxCrossEntropy:
    """Softmax activation combined with cross-entropy loss.

    This loss is used for multi-class classification. It takes raw
    network outputs, called logits, applies softmax to convert them into
    probabilities, and then computes cross-entropy against one-hot labels.

    Softmax:
        y_hat_i = exp(z_i) / sum_j exp(z_j)

    Cross-entropy:
        C = -(1 / N) * sum_n sum_i y_ni * log(y_hat_ni)

    Combined derivative:
        dC/dz = (y_hat - y) / N

    Args:
        epsilon: Small value used to avoid log(0).

    Attributes:
        logits: Cached raw network outputs from the most recent forward pass.
        y_hat: Cached softmax probabilities.
        y: Cached one-hot target labels.
        epsilon: Numerical stability constant.
    """

    def __init__(self, epsilon=1e-12):
        """Initialize the softmax cross-entropy loss.

        Args:
            epsilon: Small value used for numerical stability.
        """
        self.logits = None
        self.y_hat = None
        self.y = None
        self.epsilon = epsilon

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities.

        Args:
            logits: Raw network output scores.

        Returns:
            Probability distribution over classes.
        """
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Compute the forward pass.

        Args:
            logits: Raw network output scores.
            y: One-hot encoded target labels.

        Returns:
            Mean softmax cross-entropy loss.
        """
        self.logits = logits
        self.y = y
        self.y_hat = self.softmax(logits)

        clipped_y_hat = np.clip(self.y_hat, self.epsilon, 1.0 - self.epsilon)

        loss = -np.mean(np.sum(y * np.log(clipped_y_hat), axis=1))
        return loss

    def backprop(self) -> np.ndarray:
        """Compute the gradient with respect to the logits.

        Returns:
            Gradient with respect to the raw network outputs.

        Raises:
            ValueError: If called before `forward`.
        """
        if self.y_hat is None or self.y is None:
            raise ValueError("Forward must be called before backprop.")

        n = self.y.shape[0]
        return (self.y_hat - self.y) / n

    def predict(self) -> np.ndarray:
        """Return predicted class labels.

        Returns:
            Predicted class index for each sample.

        Raises:
            ValueError: If called before `forward`.
        """
        if self.y_hat is None:
            raise ValueError("Forward must be called before predict.")

        return np.argmax(self.y_hat, axis=1)