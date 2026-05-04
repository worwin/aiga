# aiga/tests/test_activations/test_activations.py
import numpy as np
from aiga.activations import ReLU, Abs


def test_relu():
    relu = ReLU()

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Forward pass
    expected_forward = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.allclose(relu.forward(x), expected_forward)

    # Backpropagation
    # ReLU blocks gradients where x <= 0 and passes gradients where x > 0.
    delta = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    expected_backprop = np.array([0.0, 0.0, 0.0, 40.0, 50.0])
    assert np.allclose(relu.backprop(delta), expected_backprop)


def test_abs():
    abs_act = Abs()
    x = np.array([-2, -1, 0, 1, 2])

    # Forward pass
    expected_forward = np.array([2, 1, 0, 1, 2])
    assert np.allclose(abs_act.forward(x), expected_forward)
    assert np.allclose(abs_act(x), expected_forward)

    # Derivative (sign function)
    expected_derivative = np.array([-1, -1, 0, 1, 1])
    assert np.allclose(abs_act.derivative(x), expected_derivative)

    # Backprop
    delta = np.ones_like(x)
    abs_act.forward(x)  # cache input
    expected_backprop = np.sign(x) * delta
    assert np.allclose(abs_act.backprop(delta), expected_backprop)
