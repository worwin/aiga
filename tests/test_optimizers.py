import numpy as np

from aiga.optimizers import SGD, MomentumSGD, NesterovSGD, AdaGrad, RMSProp, Adam


def test_sgd_exact_update():
    optimizer = SGD(lr=0.1)
    params = [np.array([1.0, -2.0], dtype=np.float64)]
    grads = [np.array([0.5, -1.0], dtype=np.float64)]

    before = params[0].copy()
    expected = before - 0.1 * grads[0]

    optimizer.update(params, grads)

    assert np.allclose(params[0], expected)


def test_momentum_sgd_first_and_second_update_and_velocity_state():
    optimizer = MomentumSGD(lr=0.1, gamma=0.9)
    params = [np.array([1.0], dtype=np.float64)]
    grads = [np.array([2.0], dtype=np.float64)]

    optimizer.update(params, grads)
    assert optimizer.velocities is not None
    assert np.allclose(optimizer.velocities[0], np.array([0.2]))
    assert np.allclose(params[0], np.array([0.8]))

    optimizer.update(params, grads)
    # v2 = 0.9 * 0.2 + 0.1 * 2.0 = 0.38
    assert np.allclose(optimizer.velocities[0], np.array([0.38]))
    # p2 = 0.8 - 0.38 = 0.42
    assert np.allclose(params[0], np.array([0.42]))


def test_nesterov_sgd_updates_and_stores_velocity_and_remains_finite():
    optimizer = NesterovSGD(lr=0.1, gamma=0.9)
    params = [np.array([1.0, -1.0], dtype=np.float64)]
    grads = [np.array([0.2, -0.3], dtype=np.float64)]

    before = params[0].copy()
    optimizer.update(params, grads)

    assert optimizer.velocities is not None
    assert params[0].shape == before.shape
    assert not np.allclose(params[0], before)

    for _ in range(10):
        optimizer.update(params, grads)

    assert np.all(np.isfinite(params[0]))
    assert np.all(np.isfinite(optimizer.velocities[0]))


def test_adagrad_state_initialization_growth_and_update_direction():
    optimizer = AdaGrad(lr=0.1, epsilon=1e-8)
    params = [np.array([1.0, -1.0], dtype=np.float64)]
    grads = [np.array([0.5, -0.5], dtype=np.float64)]

    before = params[0].copy()
    optimizer.update(params, grads)

    assert optimizer.accumulators is not None
    assert np.allclose(optimizer.accumulators[0], np.array([0.25, 0.25]))
    assert params[0][0] < before[0]
    assert params[0][1] > before[1]

    previous_acc = optimizer.accumulators[0].copy()
    optimizer.update(params, grads)
    assert np.all(optimizer.accumulators[0] > previous_acc)


def test_rmsprop_state_initialization_finite_repeated_updates_and_direction():
    optimizer = RMSProp(lr=0.01, beta=0.9, epsilon=1e-8)
    params = [np.array([1.0, -1.0], dtype=np.float64)]
    grads = [np.array([0.3, -0.2], dtype=np.float64)]

    before = params[0].copy()
    optimizer.update(params, grads)

    assert optimizer.averages is not None
    assert params[0][0] < before[0]
    assert params[0][1] > before[1]

    for _ in range(20):
        optimizer.update(params, grads)

    assert np.all(np.isfinite(params[0]))
    assert np.all(np.isfinite(optimizer.averages[0]))


def test_adam_state_t_increment_direction_and_finite_updates():
    optimizer = Adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params = [np.array([1.0, -1.0], dtype=np.float64)]
    grads = [np.array([0.4, -0.4], dtype=np.float64)]

    before = params[0].copy()
    optimizer.update(params, grads)

    assert optimizer.m is not None
    assert optimizer.v is not None
    assert optimizer.t == 1
    assert params[0][0] < before[0]
    assert params[0][1] > before[1]

    optimizer.update(params, grads)
    assert optimizer.t == 2

    for _ in range(20):
        optimizer.update(params, grads)

    assert np.all(np.isfinite(params[0]))
    assert np.all(np.isfinite(optimizer.m[0]))
    assert np.all(np.isfinite(optimizer.v[0]))
