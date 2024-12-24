from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    '''
    Base class for all activation functions.
    '''

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass."""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative for backpropagation."""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for the forward method"""
        return self.forward(x)