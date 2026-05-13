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


def run_momentum_demo(
    gradient_fn,
    lr=0.1,
    gamma=0.9,
    steps=20,
    initial_parameter=10.0,
):
    """Run a simple MomentumSGD simulation.

    Args:
        gradient_fn: Function returning the gradient for each step.
        lr: Learning rate.
        gamma: Momentum coefficient.
        steps: Number of update steps.
        initial_parameter: Starting parameter value.
    """
    p = initial_parameter
    v = 0.0

    print("MomentumSGD playground")
    print("Formula: v = gamma * v + lr * gradient")
    print("Formula: p = p - v")
    print()

    for step in range(steps):
        g = gradient_fn(step)
        v = gamma * v + lr * g
        p -= v

        print(
            f"Step {step:02d} | "
            f"gradient={g:8.4f} | "
            f"momentum={v:8.4f} | "
            f"parameter={p:8.4f}"
        )


def main():
    """Run the MomentumSGD playground."""
    lr = 0.1
    gamma = 0.9
    steps = 20
    initial_parameter = 10.0

    gradient_fn = constant_gradient
    # gradient_fn = decreasing_gradient
    # gradient_fn = alternating_gradient

    run_momentum_demo(
        gradient_fn=gradient_fn,
        lr=lr,
        gamma=gamma,
        steps=steps,
        initial_parameter=initial_parameter,
    )


if __name__ == "__main__":
    main()
