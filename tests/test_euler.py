import pytest
import numpy as np

from ode_solver.euler import solve_euler


def test_scalar_ode():
    """Test Euler's method on y' = y, y(0) = 1."""

    f = lambda t, y: y
    t_span = (0.0, 1.0)
    y0 = 1.0
    h = 0.1

    t_grid, y_grid = solve_euler(f, t_span, y0, h)

    # Check the number of time points and final time
    assert len(t_grid) == 11
    assert np.isclose(t_grid[-1], 1.0)

    # Check the initial condition
    assert np.isclose(y_grid[0], 1.0)

    # Exact solution at t = 1 is exp(1)
    assert np.isclose(y_grid[-1], np.exp(1.0), atol=0.2)


def test_vector_system():
    """Test Euler's method on a harmonic oscillator."""

    def f(t, y):
        return np.array([y[1], -y[0]])

    t_span = (0.0, 2 * np.pi)
    y0 = [1.0, 0.0]
    h = 0.01

    t_grid, y_grid = solve_euler(f, t_span, y0, h)

    # Check the solution shape
    assert y_grid.shape == (len(t_grid), 2)

    # Check the initial condition
    assert np.allclose(y_grid[0], [1.0, 0.0])


def test_invalid_step_size():
    """Test that a negative step size raises ValueError."""

    f = lambda t, y: y

    with pytest.raises(
        ValueError,
        match="Step size h must be positive and finite."
    ):
        solve_euler(f, (0, 1), 1.0, h=-0.1)


def test_invalid_time_span():
    """Test that an invalid time interval raises ValueError."""

    f = lambda t, y: y

    with pytest.raises(
        ValueError,
        match="Final time must be greater than initial time."
    ):
        solve_euler(f, (1, 0), 1.0, h=0.1)