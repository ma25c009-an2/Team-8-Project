import pytest
import numpy as np

from ode_solver.rk4 import solve_rk4


def test_scalar_ode():
    """Test RK4 on y' = y, y(0) = 1."""

    f = lambda t, y: y
    t_span = (0.0, 1.0)
    y0 = 1.0
    h = 0.1

    t_grid, y_grid = solve_rk4(f, t_span, y0, h)

    # Check the number of time points and final time
    assert len(t_grid) == 11
    assert np.isclose(t_grid[-1], 1.0)

    # Check the initial condition
    assert np.isclose(y_grid[0], 1.0)

    # Exact solution at t = 1 is exp(1)
    assert np.isclose(y_grid[-1], np.exp(1.0), atol=1e-5)


def test_vector_system():
    """Test RK4 on a harmonic oscillator."""

    def f(t, y):
        return np.array([y[1], -y[0]])

    t_span = (0.0, 2 * np.pi)
    y0 = [1.0, 0.0]
    h = 0.01

    t_grid, y_grid = solve_rk4(f, t_span, y0, h)

    # Check the solution shape
    assert y_grid.shape == (len(t_grid), 2)

    # Check the initial condition
    assert np.allclose(y_grid[0], [1.0, 0.0])

    # Exact solution at t = 2*pi is approximately [1, 0]
    assert np.allclose(y_grid[-1], [1.0, 0.0], atol=1e-6)


def test_invalid_step_size():
    """Test that a negative step size raises ValueError."""

    f = lambda t, y: y

    with pytest.raises(
        ValueError,
        match="Step size h must be positive and finite."
    ):
        solve_rk4(f, (0, 1), 1.0, h=-0.1)


def test_invalid_time_span():
    """Test that an invalid time interval raises ValueError."""

    f = lambda t, y: y

    with pytest.raises(
        ValueError,
        match="Final time must be greater than initial time."
    ):
        solve_rk4(f, (1, 0), 1.0, h=0.1)


def test_rk4_accuracy():
    """
    Test that RK4 gives a highly accurate result for y' = y.
    """

    f = lambda t, y: y

    t_span = (0.0, 1.0)
    y0 = 1.0
    h = 0.1

    t_grid, y_grid = solve_rk4(f, t_span, y0, h)

    exact = np.exp(1.0)
    error = abs(y_grid[-1] - exact)

    # RK4 should have a very small error for this problem
    assert error < 1e-5