"""Adaptive optimizer family.

This package contains optimizers that adapt the learning rate for each
parameter based on gradient history.

Optimizers:
    AdaGrad: Accumulates squared gradients.
    RMSProp: Uses a moving average of squared gradients.
    Adam: Combines momentum with adaptive squared-gradient scaling.
"""

from .adagrad import AdaGrad
from .rmsprop import RMSProp
from .adam import Adam
