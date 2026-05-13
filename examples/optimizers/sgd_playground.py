import numpy as np
import matplotlib.pyplot as plt

from aiga.optimizers import SGD


# This example minimizes a simple synthetic function with SGD.
# It shows the function, the known minimum, and the path SGD takes.


def quartic(x: np.ndarray) -> np.ndarray:
    """Compute y = x^4."""
    return x ** 4


def quartic_gradient(x: np.ndarray) -> np.ndarray:
    """Compute dy/dx for y = x^4."""
    return 4 * (x ** 3)


def quadratic(x: np.ndarray) -> np.ndarray:
    """Compute y = x^2."""
    return x ** 2


def quadratic_gradient(x: np.ndarray) -> np.ndarray:
    """Compute dy/dx for y = x^2."""
    return 2 * x


def run_sgd_example(
    function,
    gradient,
    learning_rate=0.001,
    steps=50,
    initial_x=2.0,
    known_minimum_x=0.0,
):
    """Run SGD on a one-variable function.

    Args:
        function: Function to minimize.
        gradient: Derivative of the function.
        learning_rate: Step-size multiplier.
        steps: Number of update steps.
        initial_x: Starting x-coordinate.
        known_minimum_x: Known minimizer used for display.

    Returns:
        History of x, y, gradient, and update values.
    """
    x = np.array([initial_x], dtype=float)
    optimizer = SGD(lr=learning_rate)

    history = {
        "x": [],
        "y": [],
        "gradient": [],
        "update": [],
    }

    print("SGD function-minimization example")
    print("Function: y = x^4")
    print(f"Known minimum: x = {known_minimum_x}, y = {function(np.array([known_minimum_x]))[0]}")
    print("Update: x = x - learning_rate * gradient")
    print()

    for step in range(steps + 1):
        y = function(x)
        grad = gradient(x)
        update = learning_rate * grad

        history["x"].append(float(x[0]))
        history["y"].append(float(y[0]))
        history["gradient"].append(float(grad[0]))
        history["update"].append(float(update[0]))

        print(
            f"Step {step:02d} | "
            f"x={x[0]:12.6f} | "
            f"y={y[0]:12.6f} | "
            f"gradient={grad[0]:12.6f} | "
            f"update={update[0]:12.6f} | "
            f"distance_to_min={abs(x[0] - known_minimum_x):12.6f}"
        )

        optimizer.update([x], [grad])

    return history


def plot_full_function(function, known_minimum_x=0.0, x_left=-3.0, x_right=3.0):
    """Plot the full function being minimized.

    Args:
        function: Function being minimized.
        known_minimum_x: Known minimizer used for display.
        x_left: Left edge of the plotted domain.
        x_right: Right edge of the plotted domain.
    """
    x_grid = np.linspace(x_left, x_right, 1000)
    y_grid = function(x_grid)
    minimum_y = function(np.array([known_minimum_x]))[0]

    plt.figure()
    plt.plot(x_grid, y_grid, label="Function: y = f(x)")
    plt.scatter([known_minimum_x], [minimum_y], marker="x", s=100, label="Known minimum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Full function being minimized")
    plt.legend()
    plt.grid(True)


def plot_sgd_path(function, history, known_minimum_x=0.0, x_left=-3.0, x_right=3.0):
    """Plot the function and the SGD path.

    Args:
        function: Function being minimized.
        history: Recorded SGD path.
        known_minimum_x: Known minimizer used for display.
        x_left: Left edge of the plotted domain.
        x_right: Right edge of the plotted domain.
    """
    x_path = np.array(history["x"])
    y_path = np.array(history["y"])

    x_grid = np.linspace(x_left, x_right, 1000)
    y_grid = function(x_grid)
    minimum_y = function(np.array([known_minimum_x]))[0]

    plt.figure()
    plt.plot(x_grid, y_grid, label="Function: y = f(x)")
    plt.scatter(x_path, y_path, label="SGD steps")
    plt.plot(x_path, y_path, linestyle="--", label="SGD path")
    plt.scatter([x_path[0]], [y_path[0]], marker="o", s=100, label="Start")
    plt.scatter([x_path[-1]], [y_path[-1]], marker="s", s=100, label="Final step")
    plt.scatter([known_minimum_x], [minimum_y], marker="x", s=100, label="Known minimum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("SGD path on the original function")
    plt.legend()
    plt.grid(True)


def main():
    """Run the SGD example."""
    function = quartic
    gradient = quartic_gradient
    known_minimum_x = 0.0

    learning_rate = 0.001
    steps = 50
    initial_x = 2.0

    # This controls how much of the original function is shown.
    x_left = -3.0
    x_right = 3.0

    # Try this simpler bowl-shaped function later.
    # function = quadratic
    # gradient = quadratic_gradient
    # learning_rate = 0.1
    # steps = 25
    # initial_x = 5.0
    # known_minimum_x = 0.0
    # x_left = -6.0
    # x_right = 6.0

    history = run_sgd_example(
        function=function,
        gradient=gradient,
        learning_rate=learning_rate,
        steps=steps,
        initial_x=initial_x,
        known_minimum_x=known_minimum_x,
    )

    plot_full_function(
        function=function,
        known_minimum_x=known_minimum_x,
        x_left=x_left,
        x_right=x_right,
    )

    plot_sgd_path(
        function=function,
        history=history,
        known_minimum_x=known_minimum_x,
        x_left=x_left,
        x_right=x_right,
    )

    plt.show()


if __name__ == "__main__":
    main()
