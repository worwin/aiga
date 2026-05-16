import numpy as np
import pytest

from aiga.layers import Dropout


def test_dropout_p_zero_returns_input_unchanged_in_training():
    layer = Dropout(p=0.0, seed=7)
    layer.train()

    x = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    out = layer.forward(x)

    assert np.allclose(out, x)


def test_dropout_training_zeros_some_activations_with_fixed_seed():
    layer = Dropout(p=0.5, seed=42)
    layer.train()

    x = np.ones((20, 20))
    out = layer.forward(x)

    assert np.any(out == 0.0)


def test_dropout_training_uses_inverted_scaling_for_kept_activations():
    p = 0.25
    keep_scale = 1.0 / (1.0 - p)
    layer = Dropout(p=p, seed=42)
    layer.train()

    x = np.ones((30, 30))
    out = layer.forward(x)

    kept = out != 0.0
    assert np.allclose(out[kept], keep_scale)


def test_dropout_backprop_uses_same_mask_and_scaling():
    p = 0.4
    keep_prob = 1.0 - p
    layer = Dropout(p=p, seed=42)
    layer.train()

    x = np.ones((10, 10))
    _ = layer.forward(x)
    delta = np.ones_like(x)
    back = layer.backprop(delta)

    expected = layer.mask / keep_prob
    assert np.allclose(back, expected)


def test_dropout_eval_mode_returns_input_unchanged():
    layer = Dropout(p=0.5, seed=42)
    layer.eval()

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = layer.forward(x)
    back = layer.backprop(np.ones_like(x))

    assert np.allclose(out, x)
    assert np.allclose(back, np.ones_like(x))


def test_dropout_invalid_p_raises_value_error():
    with pytest.raises(ValueError):
        Dropout(p=-0.1)

    with pytest.raises(ValueError):
        Dropout(p=1.0)

    with pytest.raises(ValueError):
        Dropout(p=1.1)
