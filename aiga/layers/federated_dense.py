import numpy as np
from aiga.layers.dense import Dense


class FederatedDense(Dense):
    """Dense layer with parameter-copying utilities for federated averaging.

    This layer behaves like a normal Dense layer, but adds helper methods
    for saving, restoring, and comparing parameter states.

    Attributes:
        W: Weight matrix.
        b: Bias vector.
        dW: Gradient of the loss with respect to W.
        db: Gradient of the loss with respect to b.
    """

    def get_parameters(self):
        """Return a copy of the layer parameters.

        Returns:
            Dictionary containing copied weights and bias.
        """
        return {
            "W": self.W.copy(),
            "b": self.b.copy(),
        }

    def set_parameters(self, parameters):
        """Set the layer parameters.

        Args:
            parameters: Dictionary containing weights and bias.
        """
        self.W = parameters["W"].copy()
        self.b = parameters["b"].copy()

    def get_parameter_update(self, original_parameters):
        """Compute the parameter change from an original state.

        Args:
            original_parameters: Parameters from an earlier state.

        Returns:
            Dictionary containing weight and bias changes.
        """
        return {
            "W": self.W - original_parameters["W"],
            "b": self.b - original_parameters["b"],
        }