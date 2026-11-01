import numpy as np
import pytest

from ode_solver.error import (
    max_absolute_error,
    observed_convergence_rate,
)


def test_scalar_max_absolute_error():
    numerical = np.array([1.0, 2.1, 3.0])
    exact = np.array([1.0, 2.0, 3.0])

    error = max_absolute_error(numerical, exact)

    assert np.isclose(error, 0.1)


def test_vector_max_absolute_error():
    numerical = np.array([
        [1.0, 0.0],
        [0.5, -0.8],
    ])

    exact = np.array([
        [1.0, 0.0],
        [0.5, -1.0],
    ])

    error = max_absolute_error(numerical, exact)

    assert np.isclose(error, 0.2)


def test_first_order_convergence():
    h_values = [0.2, 0.1, 0.05, 0.025]
    errors = [0.08, 0.04, 0.02, 0.01]

    rates = observed_convergence_rate(h_values, errors)

    assert np.allclose(rates, 1.0)


def test_fourth_order_convergence():
    h_values = [0.2, 0.1, 0.05, 0.025]
    errors = [0.0016, 0.0001, 0.00000625, 0.000000390625]

    rates = observed_convergence_rate(h_values, errors)

    assert np.allclose(rates, 4.0)


def test_error_shape_mismatch():
    with pytest.raises(ValueError):
        max_absolute_error([1.0, 2.0], [1.0])


def test_zero_error_rejected_for_convergence():
    with pytest.raises(ValueError):
        observed_convergence_rate(
            [0.2, 0.1],
            [0.0, 0.01],
        )
