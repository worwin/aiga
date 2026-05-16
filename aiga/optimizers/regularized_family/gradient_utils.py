"""Utilities for combining base gradients with regularization gradients."""


def apply_regularization_grads(grads, params, regularizer=None):
    """Return gradients optionally augmented with regularizer gradients.

    Args:
        grads: Iterable of base gradients.
        params: Iterable of parameters corresponding to ``grads``.
        regularizer: Regularizer instance exposing ``gradient(params)``.

    Returns:
        List of gradients. If ``regularizer`` is ``None``, this is a shallow
        list copy of ``grads``.

    Raises:
        ValueError: If gradient lengths or shapes do not match.
    """
    grads = list(grads)
    params = list(params)

    if regularizer is None:
        return grads

    regularizer_grads = list(regularizer.gradient(params))

    if len(grads) != len(regularizer_grads):
        raise ValueError("Gradient list lengths must match for regularization.")

    combined = []
    for g, rg in zip(grads, regularizer_grads):
        if g.shape != rg.shape:
            raise ValueError("Gradient shapes must match regularizer gradients.")
        combined.append(g + rg)

    return combined
