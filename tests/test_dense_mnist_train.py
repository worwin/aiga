import numpy as np

from aiga.layers.dense import Dense
from aiga.activations import ReLU
from aiga.networks.sequential import Sequential
from aiga.losses import SoftmaxCrossEntropy
from aiga.optimizers import SGD


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels into one-hot encoded labels.

    Args:
        y: Integer labels.
        num_classes: Number of classes.

    Returns:
        One-hot encoded label matrix.
    """
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        logits: Raw network output scores.
        y: Integer class labels.

    Returns:
        Classification accuracy.
    """
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == y)


def test_mnist_style_training():
    """Train a simple MNIST-style feedforward network on synthetic data.

    This test checks whether the network can run the same kind of data
    shape used by MNIST: 784 input features and 10 output classes.
    """

    np.random.seed(42)

    # MNIST-style data:
    # 64 samples, each sample has 784 features.
    x_train = np.random.randn(64, 784)

    # Fake digit labels from 0 to 9.
    y_train_labels = np.random.randint(0, 10, size=64)

    # Convert labels into one-hot vectors for cross-entropy.
    y_train = one_hot(y_train_labels, num_classes=10)

    # Build feedforward network:
    # 784 input features -> 100 hidden neurons -> 10 output scores.
    net = Sequential(
        Dense(100, input_size=784),
        ReLU(),
        Dense(10)
    )

    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(lr=0.01)

    initial_logits = net(x_train)
    initial_loss = loss_fn.forward(initial_logits, y_train)

    for _ in range(10):
        logits = net(x_train)

        loss = loss_fn.forward(logits, y_train)

        delta = loss_fn.backprop()

        net.backprop(delta)
        net.update(optimizer)

    final_logits = net(x_train)
    final_loss = loss_fn.forward(final_logits, y_train)
    final_accuracy = accuracy(final_logits, y_train_labels)

    assert final_logits.shape == (64, 10)
    assert np.isfinite(final_loss)
    assert 0.0 <= final_accuracy <= 1.0

    # This is a weak learning check, because the data is random.
    # It only checks that training does not explode.
    assert final_loss <= initial_loss or np.isclose(final_loss, initial_loss)