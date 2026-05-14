"""Regularization helpers.

This package contains regularization terms that can be added to a loss
or used to modify gradients during optimization.

Regularizers:
    L2Regularization: Penalizes squared parameter magnitude.
    TikhonovRegularization: L2/Tikhonov penalty in the identity-matrix case.
"""

from .l2_regularization import L2Regularization
from .tikhonov_regularization import TikhonovRegularization
