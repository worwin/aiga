class Abs(ActivationFunction):
    """
    Abs activation function.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.data = x  # caching input for backpropagation
        return np.abs(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)
    
    def backprop(self, delta: np.ndarray) -> np.ndarray
        if not hasattr(self, 'data'):
            raise ValueError("Forward must be called before backpropagation.")
        return np.sign(self.data) * delta
    