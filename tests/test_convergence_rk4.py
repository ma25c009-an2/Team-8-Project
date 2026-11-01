from ode_solver.rk4 import solve_rk4


def test_rk4_scalar_convergence_order():
    """
    Verify fourth-order convergence of RK4
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
        t_grid, y_grid = solve_rk4(f, t_span, y0, h)

        y_exact = exact_solution(t_grid)
        error = max_absolute_error(y_grid, y_exact)

        errors.append(error)

    rates = observed_convergence_rate(h_values, errors)

    assert np.allclose(rates, 4.0, atol=0.2)


def test_rk4_vector_convergence_order():
    """
    Verify fourth-order convergence of RK4
    for the harmonic oscillator system.
    """

    def f(t, y):
        return np.array([y[1], -y[0]])

    def exact_solution(t):
        return np.column_stack((np.cos(t), -np.sin(t)))

    t_span = (0.0, 2 * np.pi)
    y0 = np.array([1.0, 0.0])

    h_values = [
        2 * np.pi / 20,
        2 * np.pi / 40,
        2 * np.pi / 80,
        2 * np.pi / 160,
    ]

    errors = []

    for h in h_values:
        t_grid, y_grid = solve_rk4(f, t_span, y0, h)

        y_exact = exact_solution(t_grid)
        error = max_absolute_error(y_grid, y_exact)

        errors.append(error)

    rates = observed_convergence_rate(h_values, errors)

    assert np.allclose(rates, 4.0, atol=0.2)
