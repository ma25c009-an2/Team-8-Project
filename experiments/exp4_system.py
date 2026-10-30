import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ode_solver.euler import solve_euler


# Harmonic oscillator:
# y1' = y2
# y2' = -y1

def f(t, y):
    return np.array([y[1], -y[0]])


# Initial condition and time interval
y0 = np.array([1.0, 0.0])
t_span = (0.0, 2 * np.pi)
h = 0.1

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# Euler method using the team's existing function
t_euler, y_euler = solve_euler(f, t_span, y0, h)


# RK4 method
def solve_rk4_local(f, t_span, y0, h):
    t0, tf = t_span
    n = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)

    y = np.zeros((n + 1, len(y0)))
    y[0] = y0

    for i in range(n):
        step = t[i + 1] - t[i]
        ti = t[i]
        yi = y[i]

        k1 = f(ti, yi)
        k2 = f(ti + step / 2, yi + step * k1 / 2)
        k3 = f(ti + step / 2, yi + step * k2 / 2)
        k4 = f(ti + step, yi + step * k3)

        y[i + 1] = yi + (step / 6) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

    return t, y


t_rk4, y_rk4 = solve_rk4_local(f, t_span, y0, h)


# Exact solution
t_exact = np.linspace(0, 2 * np.pi, 1000)
exact_y1 = np.cos(t_exact)
exact_y2 = -np.sin(t_exact)


# Plot the phase-plane trajectories
plt.figure(figsize=(8, 8))

plt.plot(
    exact_y1, exact_y2,
    "--", label="Exact unit circle"
)

plt.plot(
    y_euler[:, 0], y_euler[:, 1],
    "o-", markersize=2, label="Euler"
)

plt.plot(
    y_rk4[:, 0], y_rk4[:, 1],
    "s-", markersize=2, label="RK4"
)

plt.xlabel("y1")
plt.ylabel("y2")
plt.title("Experiment 4: Harmonic Oscillator")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(RESULTS / "exp4_system.png", dpi=200)
plt.close()


# Save numerical results
with open(RESULTS / "exp4_system.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "time",
        "Euler_y1", "Euler_y2", "Euler_radius",
        "RK4_y1", "RK4_y2", "RK4_radius",
        "Exact_y1", "Exact_y2", "Exact_radius"
    ])

    for i in range(len(t_euler)):
        ti = t_euler[i]

        j = np.argmin(np.abs(t_rk4 - ti))

        writer.writerow([
            ti,
            y_euler[i, 0],
            y_euler[i, 1],
            np.linalg.norm(y_euler[i]),
            y_rk4[j, 0],
            y_rk4[j, 1],
            np.linalg.norm(y_rk4[j]),
            np.cos(ti),
            -np.sin(ti),
            1.0
        ])


# Save final-radius summary
with open(
    RESULTS / "exp4_system_summary.csv", "w", newline=""
) as file:
    writer = csv.writer(file)

    writer.writerow(["Method", "Step size", "Final radius"])
    writer.writerow([
        "Euler", h, np.linalg.norm(y_euler[-1])
    ])
    writer.writerow([
        "RK4", h, np.linalg.norm(y_rk4[-1])
    ])
    writer.writerow(["Exact", h, 1.0])


print("Experiment 4 completed.")
print("Euler final radius:", np.linalg.norm(y_euler[-1]))
print("RK4 final radius:", np.linalg.norm(y_rk4[-1]))
print("Exact radius: 1.0")
