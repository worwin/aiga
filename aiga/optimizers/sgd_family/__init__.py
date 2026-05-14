"""SGD optimizer family.

This package contains optimizers based on stochastic gradient descent.

Optimizers:
    SGD: Plain stochastic gradient descent.
    MomentumSGD: SGD with momentum.
    NesterovSGD: SGD with Nesterov accelerated gradient.
"""

from .sgd import SGD
from .momentum_sgd import MomentumSGD
from .nesterov_sgd import NesterovSGD
