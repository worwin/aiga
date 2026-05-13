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


def run_adam_demo(
    gradient_fn,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    steps=20,
    initial_parameter=10.0,
):
    """Run a simple Adam simulation.

    Args:
        gradient_fn: Function returning the gradient for each step.
        lr: Learning rate.
        beta1: First moment decay rate.
        beta2: Second moment decay rate.
        epsilon: Numerical stability constant.
        steps: Number of update steps.
        initial_parameter: Starting parameter value.
    """
    p = initial_parameter
    m = 0.0
    v = 0.0

    print("Adam playground")
    print("Formula: m = beta1 * m + (1 - beta1) * gradient")
    print("Formula: v = beta2 * v + (1 - beta2) * gradient^2")
    print("Formula: p = p - lr * m_hat / (sqrt(v_hat) + epsilon)")
    print()

    for step in range(1, steps + 1):
        g = gradient_fn(step - 1)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)

        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)

        update = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        p -= update

        print(
            f"Step {step:02d} | "
            f"gradient={g:8.4f} | "
            f"m={m:8.4f} | "
            f"v={v:8.4f} | "
            f"update={update:8.4f} | "
            f"parameter={p:8.4f}"
        )


def main():
    """Run the Adam playground."""
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    steps = 20
    initial_parameter = 10.0

    gradient_fn = constant_gradient
    # gradient_fn = decreasing_gradient
    # gradient_fn = alternating_gradient

    run_adam_demo(
        gradient_fn=gradient_fn,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        steps=steps,
        initial_parameter=initial_parameter,
    )


if __name__ == "__main__":
    main()
