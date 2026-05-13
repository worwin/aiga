import numpy as np


def constant_gradient(step):
    """Return a constant gradient."""
    return 5.0


def decreasing_gradient(step):
    """Return a gradually decreasing gradient."""
    return max(5.0 - 0.25 * step, 0.0)


def alternating_gradient(step):
    """Return a gradient that changes direction."""
    return 5.0 if step < 10 else -5.0


def run_sgd_demo(gradient_fn, lr=0.1, steps=20, initial_parameter=10.0):
    """Run a simple SGD simulation.

    Args:
        gradient_fn: Function returning the gradient for each step.
        lr: Learning rate.
        steps: Number of update steps.
        initial_parameter: Starting parameter value.
    """
    p = initial_parameter

    print("Plain SGD playground")
    print("Formula: p = p - lr * gradient")
    print()

    for step in range(steps):
        g = gradient_fn(step)
        update = lr * g
        p -= update

        print(
            f"Step {step:02d} | "
            f"gradient={g:8.4f} | "
            f"update={update:8.4f} | "
            f"parameter={p:8.4f}"
        )


def main():
    """Run the SGD playground."""
    lr = 0.1
    steps = 20
    initial_parameter = 10.0

    gradient_fn = constant_gradient
    # gradient_fn = decreasing_gradient
    # gradient_fn = alternating_gradient

    run_sgd_demo(
        gradient_fn=gradient_fn,
        lr=lr,
        steps=steps,
        initial_parameter=initial_parameter,
    )


if __name__ == "__main__":
    main()
