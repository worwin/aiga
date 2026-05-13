import numpy as np


# Choose one objective function and its derivative.
# The optimizer will move x in the direction that lowers y = f(x).

def quadratic(x):
    """Compute y = x^2."""
    return x ** 2


def quadratic_gradient(x):
    """Compute dy/dx for y = x^2."""
    return 2 * x


def cubic(x):
    """Compute y = x^3."""
    return x ** 3


def cubic_gradient(x):
    """Compute dy/dx for y = x^3."""
    return 3 * (x ** 2)


def run_sgd_playground(
    function,
    gradient,
    learning_rate=0.1,
    steps=20,
    initial_x=5.0,
):
    """Run SGD on a simple one-variable function.

    Args:
        function: Function to minimize.
        gradient: Derivative of the function.
        learning_rate: Step-size multiplier.
        steps: Number of SGD steps.
        initial_x: Starting x-coordinate.
    """
    x = initial_x

    print("SGD playground")
    print("Formula: x = x - learning_rate * gradient")
    print()

    for step in range(steps + 1):
        y = function(x)
        grad = gradient(x)
        update = learning_rate * grad

        print(
            f"Step {step:02d} | "
            f"x={x:10.5f} | "
            f"y={y:10.5f} | "
            f"gradient={grad:10.5f} | "
            f"update={update:10.5f}"
        )

        x = x - update


def main():
    """Run the SGD playground."""
    learning_rate = 0.1
    steps = 20
    initial_x = 5.0

    function = quadratic
    gradient = quadratic_gradient

    # Try this later, but use a smaller learning rate.
    # function = cubic
    # gradient = cubic_gradient
    # learning_rate = 0.001

    run_sgd_playground(
        function=function,
        gradient=gradient,
        learning_rate=learning_rate,
        steps=steps,
        initial_x=initial_x,
    )


if __name__ == "__main__":
    main()
