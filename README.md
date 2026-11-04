# Team-8-Project
ODE SOLVER AND STABILITY EXPLORER
## Project Overview

This project is part of the Scientific Computing course.

The objective is to implement numerical methods for solving initial-value problems of ordinary differential equations (ODEs) and study their accuracy, convergence, and stability.

We implement Euler's method and the classical fourth-order Runge-Kutta method (RK4).

## Objectives

- Implement Euler's method.
- Implement the classical RK4 method.
- Solve scalar ODEs and systems of ODEs.
- Compare numerical solutions with exact solutions.
- Calculate maximum absolute error.
- Study convergence for different step sizes.
- Investigate the stability of Euler's method.

## Project Structure

- `docs/` — Project documentation.
- `experiments/` — Numerical experiment scripts.
- `ode_solver/` — Euler, RK4, and error calculation implementations.
- `report/` — Project report.
- `result/` — Numerical results and plots.
- `tests/` — Unit tests.

## Team Contributions

- **Member 1:** Team Leader and Euler's Method.
- **Member 2:** Runge-Kutta Method (RK4).
- **Member 3:** Error and Convergence Analysis.
- **Member 4:** Stability Experiments and Visualization.

## Installation

Install the required packages:

```bash
python -m pip install numpy matplotlib pytest
```

## Running Tests

```bash
python -m pytest
```

## Conclusion

This project demonstrates numerical methods for solving ODEs and investigates their accuracy, convergence, and stability through computational experiments.
