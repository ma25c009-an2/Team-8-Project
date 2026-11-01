import numpy as np


def max_absolute_error(y_numerical, y_exact):
    """
    Calculate the maximum absolute error between
    numerical and exact solutions.

    Supports scalar and vector-valued ODEs.
    """
    y_numerical = np.asarray(y_numerical, dtype=float)
    y_exact = np.asarray(y_exact, dtype=float)

    if y_numerical.shape != y_exact.shape:
        raise ValueError(
            "Numerical and exact solutions must have the same shape."
        )

    if not np.all(np.isfinite(y_numerical)):
        raise ValueError("Numerical solution must contain finite values.")

    if not np.all(np.isfinite(y_exact)):
        raise ValueError("Exact solution must contain finite values.")

    return float(np.max(np.abs(y_numerical - y_exact)))


def observed_convergence_rate(h_values, errors):
    """
    Calculate observed convergence rates between
    successive step sizes.
    """
    h_values = np.asarray(h_values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if h_values.ndim != 1 or errors.ndim != 1:
        raise ValueError("Inputs must be one-dimensional arrays.")

    if len(h_values) != len(errors) or len(h_values) < 2:
        raise ValueError("Need at least two matching step sizes and errors.")

    if np.any(h_values <= 0) or np.any(errors <= 0):
        raise ValueError("Step sizes and errors must be positive.")

    if np.any(np.diff(h_values) >= 0):
        raise ValueError("Step sizes must be strictly decreasing.")

    return (
        np.log(errors[:-1] / errors[1:])
        / np.log(h_values[:-1] / h_values[1:])
    )
