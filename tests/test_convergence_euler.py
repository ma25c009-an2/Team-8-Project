import numpy as np

from ode_solver.euler import solve_euler
from ode_solver.error import (
    max_absolute_error,
    observed_convergence_rate,
)


def test_euler_scalar_convergence_order():
    """
    Verify first-order convergence of Euler's method
    for the scalar IVP y' = y, y(0) = 1.
    """

    def f(t, y):
        return y

    def exact_solution(t):
        return np.exp(t)

    t_span = (0.0, 1.0)
    y0 = 1.0

    h_values = [0.2, 0.1, 0.05, 0.025]
    errors = []

    for h in h_values:
        t_grid, y_grid = solve_euler(f, t_span, y0, h)

        y_exact = exact_solution(t_grid)
        error = max_absolute_error(y_grid, y_exact)

        errors.append(error)

    rates = observed_convergence_rate(h_values, errors)

    assert np.allclose(rates, 1.0, atol=0.15)


def test_euler_vector_convergence_order():
    """
    Verify first-order convergence of Euler's method
    for the harmonic oscillator system.
    """

    def f(t, y):
        return np.array([y[1], -y[0]])

    def exact_solution(t):
        return np.column_stack((np.cos(t), -np.sin(t)))

    t_span = (0.0, 2 * np.pi)
    y0 = np.array([1.0, 0.0])

    # These step sizes divide the interval into integer numbers of steps.
    h_values = [
        2 * np.pi / 20,
        2 * np.pi / 40,
        2 * np.pi / 80,
        2 * np.pi / 160,
    ]

    errors = []

    for h in h_values:
        t_grid, y_grid = solve_euler(f, t_span, y0, h)

        y_exact = exact_solution(t_grid)
        error = max_absolute_error(y_grid, y_exact)

        errors.append(error)

    rates = observed_convergence_rate(h_values, errors)

    assert np.allclose(rates, 1.0, atol=0.15)
