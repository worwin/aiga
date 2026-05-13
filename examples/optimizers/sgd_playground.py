import numpy as np

from aiga.optimizers import SGD


# This example minimizes a simple synthetic function with SGD.
# It is intentionally small so the parameter movement is easy to inspect.


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
    steps=25,
    initial_x=2.0,
):
    """Run SGD on a one-variable function.

    Args:
        function: Function to minimize.
        gradient: Derivative of the function.
        learning_rate: Step-size multiplier.
        steps: Number of update steps.
        initial_x: Starting x-coordinate.
    """
    x = np.array([initial_x], dtype=float)
    optimizer = SGD(lr=learning_rate)

    print("SGD function-minimization example")
    print("Goal: minimize y = f(x)")
    print("Update: x = x - learning_rate * gradient")
    print()

    for step in range(steps + 1):
        y = function(x)
        grad = gradient(x)
        update = learning_rate * grad

        print(
            f"Step {step:02d} | "
            f"x={x[0]:12.6f} | "
            f"y={y[0]:12.6f} | "
            f"gradient={grad[0]:12.6f} | "
            f"update={update[0]:12.6f}"
        )

        optimizer.update([x], [grad])


def main():
    """Run the SGD example."""
    function = quartic
    gradient = quartic_gradient

    learning_rate = 0.001
    steps = 25
    initial_x = 2.0

    # Try this simpler bowl-shaped function later.
    # function = quadratic
    # gradient = quadratic_gradient
    # learning_rate = 0.1
    # initial_x = 5.0

    run_sgd_example(
        function=function,
        gradient=gradient,
        learning_rate=learning_rate,
        steps=steps,
        initial_x=initial_x,
    )


if __name__ == "__main__":
    main()
