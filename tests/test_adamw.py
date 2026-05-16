import numpy as np

from aiga.optimizers import Adam, AdamW


def test_adamw_initializes_without_error():
    optimizer = AdamW()
    assert optimizer is not None
    assert optimizer.t == 0
    assert optimizer.m is None
    assert optimizer.v is None


def test_adamw_first_update_changes_parameters_in_expected_direction():
    params = [np.array([1.0, -1.0], dtype=np.float64)]
    grads = [np.array([0.5, -0.5], dtype=np.float64)]

    optimizer = AdamW(lr=0.1, weight_decay=0.0)
    before = params[0].copy()
    optimizer.update(params, grads)

    # Positive gradient should push parameter down; negative gradient up.
    assert params[0][0] < before[0]
    assert params[0][1] > before[1]


def test_adamw_creates_internal_state_and_increments_t():
    params = [np.array([1.0, 2.0], dtype=np.float64)]
    grads = [np.array([0.1, 0.2], dtype=np.float64)]
    optimizer = AdamW()

    optimizer.update(params, grads)
    assert optimizer.m is not None
    assert optimizer.v is not None
    assert len(optimizer.m) == 1
    assert len(optimizer.v) == 1
    assert optimizer.t == 1

    optimizer.update(params, grads)
    assert optimizer.t == 2


def test_adamw_zero_weight_decay_matches_adam_first_step():
    params_adam = [np.array([1.0, -2.0], dtype=np.float64)]
    params_adamw = [params_adam[0].copy()]
    grads = [np.array([0.3, -0.4], dtype=np.float64)]

    adam = Adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    adamw = AdamW(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)

    adam.update(params_adam, grads)
    adamw.update(params_adamw, grads)

    assert np.allclose(params_adam[0], params_adamw[0])


def test_adamw_weight_decay_shrinks_parameters_more_than_adam():
    params_adam = [np.array([1.0, -2.0], dtype=np.float64)]
    params_adamw = [params_adam[0].copy()]
    grads = [np.array([0.2, -0.2], dtype=np.float64)]

    adam = Adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    adamw = AdamW(
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.1,
    )

    adam.update(params_adam, grads)
    adamw.update(params_adamw, grads)

    assert np.abs(params_adamw[0][0]) < np.abs(params_adam[0][0])
    assert np.abs(params_adamw[0][1]) < np.abs(params_adam[0][1])
