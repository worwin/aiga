"""Second-order optimizer family.

This package is reserved for methods that use curvature information,
approximations to curvature, or Hessian-like structure.

Planned optimizers:
    Newton: Uses Hessian information directly.
    BFGS: Builds an approximation to inverse curvature.
    L-BFGS: Limited-memory BFGS for larger parameter spaces.
"""
