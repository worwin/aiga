import numpy as np

from aiga.layers import FederatedDense
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
    """Load and normalize MNIST."""
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype(float) / 255.0
    x_test = x_test.reshape(-1, 784).astype(float) / 255.0

    return x_train, y_train, x_test, y_test


def get_model_parameters(net: Sequential) -> list[dict[str, np.ndarray]]:
    """Copy parameters from all trainable federated layers."""
    parameters = []

    for layer in net.layers:
        if layer.trainable:
            parameters.append(layer.get_parameters())

    return parameters


def set_model_parameters(net: Sequential, parameters: list[dict[str, np.ndarray]]) -> None:
    """Set parameters for all trainable federated layers."""
    parameter_index = 0

    for layer in net.layers:
        if layer.trainable:
            layer.set_parameters(parameters[parameter_index])
            parameter_index += 1


def average_parameter_sets(
    parameter_sets: list[list[dict[str, np.ndarray]]]
) -> list[dict[str, np.ndarray]]:
    """Average several locally trained parameter sets."""
    averaged_parameters = []

    num_clients = len(parameter_sets)
    num_layers = len(parameter_sets[0])

    for layer_index in range(num_layers):
        avg_W = sum(params[layer_index]["W"] for params in parameter_sets) / num_clients
        avg_b = sum(params[layer_index]["b"] for params in parameter_sets) / num_clients

        averaged_parameters.append({
            "W": avg_W,
            "b": avg_b,
        })

    return averaged_parameters


def train_local_client(
    net: Sequential,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    local_steps: int,
    learning_rate: float,
) -> float:
    """Train one local client model on one batch."""
    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(lr=learning_rate)

    loss = 0.0

    for _ in range(local_steps):
        logits = net(x_batch)

        loss = loss_fn.forward(logits, y_batch)

        delta = loss_fn.backprop()

        net.backprop(delta)
        net.update(optimizer)

    return loss


def main():
    """Run federated averaging on MNIST."""
    np.random.seed(42)

    x_train, y_train_labels, x_test, y_test_labels = load_mnist_from_keras()
    y_train = one_hot(y_train_labels, num_classes=10)

    net = Sequential(
        FederatedDense(100, input_size=784),
        ReLU(),
        FederatedDense(10, input_size=100),
    )

    rounds = 20
    num_clients = 10
    local_steps = 5
    batch_size = 256
    learning_rate = 0.1

    for round_index in range(rounds):
        global_parameters = get_model_parameters(net)
        local_parameter_sets = []
        local_losses = []

        for client_index in range(num_clients):
            batch_indices = np.random.choice(
                x_train.shape[0],
                size=batch_size,
                replace=False,
            )

            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Every client starts from the same global model.
            set_model_parameters(net, global_parameters)

            local_loss = train_local_client(
                net=net,
                x_batch=x_batch,
                y_batch=y_batch,
                local_steps=local_steps,
                learning_rate=learning_rate,
            )

            local_losses.append(local_loss)

            # Save where this client moved the model.
            local_parameter_sets.append(get_model_parameters(net))

        averaged_parameters = average_parameter_sets(local_parameter_sets)

        # Update the global model with the averaged local destinations.
        set_model_parameters(net, averaged_parameters)

        train_logits = net(x_train[:5000])
        test_logits = net(x_test)

        train_acc = accuracy(train_logits, y_train_labels[:5000])
        test_acc = accuracy(test_logits, y_test_labels)

        mean_local_loss = np.mean(local_losses)

        print(
            f"Round {round_index:03d} | "
            f"Local loss: {mean_local_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Test acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()