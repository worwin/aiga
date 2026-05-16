import numpy as np

from aiga.optimizers import SGD, L2Regularization, apply_regularization_grads


def test_apply_regularization_grads_adds_l2_gradients_correctly():
    grads = [np.array([1.0, -2.0], dtype=np.float64)]
    params = [np.array([3.0, -4.0], dtype=np.float64)]
    regularizer = L2Regularization(lambda_=0.1)

    combined = apply_regularization_grads(grads, params, regularizer=regularizer)
    expected = grads[0] + 2.0 * 0.1 * params[0]

    assert len(combined) == 1
    assert np.allclose(combined[0], expected)


def test_apply_regularization_grads_none_keeps_existing_behavior_unchanged():
    optimizer = SGD(lr=0.1)
    grads = [np.array([0.5, -0.25], dtype=np.float64)]

    params_plain = [np.array([1.0, -1.0], dtype=np.float64)]
    params_via_helper = [params_plain[0].copy()]

    optimizer.update(params_plain, grads)

    combined = apply_regularization_grads(grads, params_via_helper, regularizer=None)
    optimizer.update(params_via_helper, combined)

    assert np.allclose(params_plain[0], params_via_helper[0])


def test_l2_regularizer_changes_update_compared_to_no_regularizer():
    optimizer = SGD(lr=0.1)
    regularizer = L2Regularization(lambda_=0.2)

    grads = [np.array([0.5, -0.25], dtype=np.float64)]
    params_no_reg = [np.array([1.0, -1.0], dtype=np.float64)]
    params_with_reg = [params_no_reg[0].copy()]

    optimizer.update(params_no_reg, grads)

    combined = apply_regularization_grads(grads, params_with_reg, regularizer=regularizer)
    optimizer.update(params_with_reg, combined)

    assert not np.allclose(params_no_reg[0], params_with_reg[0])
    assert np.abs(params_with_reg[0][0]) < np.abs(params_no_reg[0][0])
    assert np.abs(params_with_reg[0][1]) < np.abs(params_no_reg[0][1])


def test_apply_regularization_grads_preserves_shapes():
    grads = [
        np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64),
        np.array([0.5, -0.6], dtype=np.float64),
    ]
    params = [
        np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
        np.array([1.5, -2.5], dtype=np.float64),
    ]
    regularizer = L2Regularization(lambda_=0.01)

    combined = apply_regularization_grads(grads, params, regularizer=regularizer)

    assert len(combined) == len(grads)
    for c, g in zip(combined, grads):
        assert c.shape == g.shape
