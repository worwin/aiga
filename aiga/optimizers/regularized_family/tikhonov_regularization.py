from .l2_regularization import L2Regularization


class TikhonovRegularization(L2Regularization):
    """Tikhonov regularization.

    In the simple identity-matrix case, Tikhonov regularization is the
    same as L2 regularization. It penalizes large parameter values by
    adding a squared-parameter term to the objective function.

    Formula:
        R(W) = lambda * ||W||_2^2

    Derivative:
        dR/dW = 2 * lambda * W

    Args:
        lambda_: Regularization strength.
    """

    def __init__(self, lambda_=0.01):
        """Initialize Tikhonov regularization.

        Args:
            lambda_: Regularization strength.
        """
        super().__init__(lambda_=lambda_)
