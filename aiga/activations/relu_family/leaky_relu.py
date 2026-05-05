import numpy as np
from aiga.layers.layer import Layer


class LeakyReLU(Layer):
    """Leaky Rectified Linear Unit activation function.

    Leaky ReLU is a modified version of ReLU that allows a small,
    non-zero slope for inputs less than or equal to zero. This helps
    prevent neurons from becoming inactive during training.

    Formula:
        f(x) = alpha * x, if x <= 0
        f(x) = x,         if x > 0

    Derivative:
        f'(x) = alpha, if x <= 0
        f'(x) = 1,     if x > 0

    Args:
        alpha: Slope applied to inputs less than or equal to zero.
            Defaults to 0.01.

    Attributes:
        x: Cached input from the most recent forward pass.
        alpha: Negative-side slope parameter.
    """

    def __init__(self, alpha=0.01):
        """Initialize the Leaky ReLU activation function.

        Args:
            alpha: Slope applied to inputs less than or equal to zero.
        """
        super().__init__()
        self.x = None
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for the forward pass.

        Args:
            x: Input array.

        Returns:
            Activated output.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass.

        Args:
            x: Input array.

        Returns:
            Activated output.
        """
        self.x = x
        return np.where(x <= 0, self.alpha * x, x)

    def derivative(self) -> np.ndarray:
        """Compute the local derivative.

        Returns:
            Elementwise derivative.

        Raises:
            ValueError: If called before `forward`.
        """
        if self.x is None:
            raise ValueError("Forward must be called before derivative.")

        derivative = np.where(self.x <= 0, self.alpha, 1.0)
        return derivative

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """Backpropagate the incoming gradient.

        Args:
            delta: Incoming gradient from the next layer.

        Returns:
            Backpropagated gradient.
        """
        return self.derivative() * delta