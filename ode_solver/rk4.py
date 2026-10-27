import numpy as np


def solve_rk4(f, t_span, y0, h):
    """
    Solve the initial-value problem y' = f(t, y)
    using the classical fourth-order Runge-Kutta (RK4) method.

    Parameters:
        f      : Function that returns the derivative f(t, y).
        t_span : Tuple (t0, tf) containing the initial and final times.
        y0     : Initial value (scalar or vector).
        h      : Positive step size.

    Returns:
        t_grid : Array of time points.
        y_grid : Numerical solution at each time point.

    Order of accuracy:
        Local truncation error  : O(h^5)
        Global truncation error : O(h^4)
    """

    t0, tf = t_span

    # Step 2: Validate the inputs
    if not callable(f):
        raise TypeError("f must be a callable function.")

    if not np.isfinite(t0) or not np.isfinite(tf):
        raise ValueError("Time values must be finite.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("Step size h must be positive and finite.")

    if tf <= t0:
        raise ValueError("Final time must be greater than initial time.")

    # Step 3: Prepare the initial condition
    y0_array = np.asarray(y0, dtype=float)

    if y0_array.ndim > 1 or y0_array.size == 0:
        raise ValueError("y0 must be a scalar or a non-empty vector.")

    if not np.all(np.isfinite(y0_array)):
        raise ValueError("Initial values must be finite.")

    is_scalar = y0_array.ndim == 0
    y0_array = np.atleast_1d(y0_array)

    # Step 4: Create the time grid
    interval = tf - t0
    num_steps = int(round(interval / h))

    if np.isclose(num_steps * h, interval, rtol=1e-12, atol=1e-12):
        t_grid = t0 + h * np.arange(num_steps + 1)
        t_grid[-1] = tf
    else:
        num_full_steps = int(np.floor(interval / h))
        t_grid = t0 + h * np.arange(num_full_steps + 1)

        if t_grid[-1] < tf:
            t_grid = np.append(t_grid, tf)

    if len(t_grid) > 1 and np.any(np.diff(t_grid) <= 0):
        raise ValueError("Step size is too small for the time values.")

    # Step 5: Store the numerical solution
    y_grid = np.zeros((len(t_grid), len(y0_array)))
    y_grid[0] = y0_array

    # Step 6: Apply classical RK4 at each time step
    for i in range(len(t_grid) - 1):

        # Length of the current step
        step = t_grid[i + 1] - t_grid[i]

        # Current time and numerical solution
        t_current = t_grid[i]

        if is_scalar:
            y_current = y_grid[i, 0]
        else:
            y_current = y_grid[i]

        # RK4 stage 1
        k1 = np.asarray(
            f(t_current, y_current),
            dtype=float
        )
        k1 = np.atleast_1d(k1)

        # RK4 stage 2
        k2 = np.asarray(
            f(
                t_current + step / 2,
                y_current + (step / 2) * k1
            ),
            dtype=float
        )
        k2 = np.atleast_1d(k2)

        # RK4 stage 3
        k3 = np.asarray(
            f(
                t_current + step / 2,
                y_current + (step / 2) * k2
            ),
            dtype=float
        )
        k3 = np.atleast_1d(k3)

        # RK4 stage 4
        k4 = np.asarray(
            f(
                t_current + step,
                y_current + step * k3
            ),
            dtype=float
        )
        k4 = np.atleast_1d(k4)

        # Check all RK4 stages
        for k in (k1, k2, k3, k4):
            if k.shape != y0_array.shape:
                raise ValueError(
                    "The derivative must have the same shape as y0."
                )

            if not np.all(np.isfinite(k)):
                raise ValueError(
                    "The derivative must contain only finite values."
                )

        # Classical RK4 update formula:
        #
        # y_(n+1) = y_n + h/6 *
        #           (k1 + 2*k2 + 2*k3 + k4)
        #
        y_grid[i + 1] = (
            y_grid[i]
            + (step / 6.0)
            * (k1 + 2 * k2 + 2 * k3 + k4)
        )

    # Step 7: Return the solution in the appropriate format
    if is_scalar:
        return t_grid, y_grid[:, 0]

    return t_grid, y_grid