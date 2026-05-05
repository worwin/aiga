import numpy as np

from aiga.layers.dense import Dense
from aiga.activations import ReLU
from aiga.networks.sequential import Sequential
from aiga.losses import SoftmaxCrossEntropy
from aiga.optimizers import SGD


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels into one-hot encoded labels."""
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy."""
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == y)


def load_mnist_from_keras():
    """Load MNIST using Keras.

    Returns:
        Normalized train and test arrays.
    """
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype(float) / 255.0
    x_test = x_test.reshape(-1, 784).astype(float) / 255.0

    return x_train, y_train, x_test, y_test


def main():
    """Train a simple feedforward network on MNIST."""

    np.random.seed(42)

    x_train, y_train_labels, x_test, y_test_labels = load_mnist_from_keras()

    # Keep this small at first so it runs quickly.
    x_train = x_train[:1000]
    y_train_labels = y_train_labels[:1000]

    x_test = x_test[:200]
    y_test_labels = y_test_labels[:200]

    y_train = one_hot(y_train_labels, num_classes=10)

    net = Sequential(
        Dense(100, input_size=784),
        ReLU(),
        Dense(10)
    )

    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(lr=0.1)

    epochs = 20

    for epoch in range(epochs):
        logits = net(x_train)

        loss = loss_fn.forward(logits, y_train)

        delta = loss_fn.backprop()

        net.backprop(delta)
        net.update(optimizer)

        train_acc = accuracy(logits, y_train_labels)
        test_logits = net(x_test)
        test_acc = accuracy(test_logits, y_test_labels)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Test acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()