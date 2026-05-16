import numpy as np

from aiga.optimizers import L2Regularization, TikhonovRegularization


def test_l2_penalty_single_parameter_array():
    regularizer = L2Regularization(lambda_=0.1)
    params = [np.array([1.0, -2.0, 3.0], dtype=np.float64)]

    expected = 0.1 * (1.0**2 + (-2.0) ** 2 + 3.0**2)
    actual = regularizer.penalty(params)

    assert np.allclose(actual, expected)


def test_l2_penalty_multiple_parameter_arrays():
    regularizer = L2Regularization(lambda_=0.25)
    params = [
        np.array([1.0, -2.0], dtype=np.float64),
        np.array([[0.5, -1.5], [2.0, -0.5]], dtype=np.float64),
    ]

    expected = 0.25 * (
        (1.0**2 + (-2.0) ** 2)
        + (0.5**2 + (-1.5) ** 2 + 2.0**2 + (-0.5) ** 2)
    )
    actual = regularizer.penalty(params)

    assert np.allclose(actual, expected)


def test_l2_gradient_matches_two_lambda_times_parameter():
    lambda_ = 0.2
    regularizer = L2Regularization(lambda_=lambda_)
    params = [
        np.array([1.0, -2.0], dtype=np.float64),
        np.array([[3.0, -4.0]], dtype=np.float64),
    ]

    grads = regularizer.gradient(params)

    assert len(grads) == len(params)
    assert np.allclose(grads[0], 2.0 * lambda_ * params[0])
    assert np.allclose(grads[1], 2.0 * lambda_ * params[1])


def test_l2_gradient_preserves_parameter_shapes():
    regularizer = L2Regularization(lambda_=0.01)
    params = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    ]

    grads = regularizer.gradient(params)

    assert grads[0].shape == params[0].shape
    assert grads[1].shape == params[1].shape


def test_l2_zero_lambda_gives_zero_penalty():
    regularizer = L2Regularization(lambda_=0.0)
    params = [np.array([1.0, -2.0, 3.0], dtype=np.float64)]

    assert np.allclose(regularizer.penalty(params), 0.0)


def test_l2_zero_lambda_gives_zero_gradients():
    regularizer = L2Regularization(lambda_=0.0)
    params = [
        np.array([1.0, -2.0], dtype=np.float64),
        np.array([[3.0, -4.0]], dtype=np.float64),
    ]

    grads = regularizer.gradient(params)

    assert np.allclose(grads[0], np.zeros_like(params[0]))
    assert np.allclose(grads[1], np.zeros_like(params[1]))


def test_tikhonov_penalty_matches_l2_penalty():
    params = [
        np.array([1.0, -2.0], dtype=np.float64),
        np.array([[0.5, -1.5]], dtype=np.float64),
    ]
    lambda_ = 0.05
    l2 = L2Regularization(lambda_=lambda_)
    tikhonov = TikhonovRegularization(lambda_=lambda_)

    assert np.allclose(l2.penalty(params), tikhonov.penalty(params))


def test_tikhonov_gradients_match_l2_gradients():
    params = [
        np.array([1.0, -2.0], dtype=np.float64),
        np.array([[0.5, -1.5]], dtype=np.float64),
    ]
    lambda_ = 0.05
    l2 = L2Regularization(lambda_=lambda_)
    tikhonov = TikhonovRegularization(lambda_=lambda_)

    l2_grads = l2.gradient(params)
    tik_grads = tikhonov.gradient(params)

    assert len(l2_grads) == len(tik_grads)
    assert np.allclose(l2_grads[0], tik_grads[0])
    assert np.allclose(l2_grads[1], tik_grads[1])
