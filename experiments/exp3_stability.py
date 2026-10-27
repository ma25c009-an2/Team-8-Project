import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ode_solver.euler import solve_euler


# Problem: y' = -lambda*y, y(0) = 1
lam = 10.0
y0 = 1.0
t_span = (0.0, 1.5)

# Step sizes for three stability behaviours
step_sizes = [0.05, 0.15, 0.25]

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def f(t, y):
    return -lam * y


# Exact solution
t_exact = np.linspace(t_span[0], t_span[1], 1000)
y_exact = y0 * np.exp(-lam * t_exact)

plt.figure(figsize=(10, 6))
plt.plot(t_exact, y_exact, label="Exact solution", linewidth=2)

summary = []
rows = []

for h in step_sizes:
    t, y = solve_euler(f, t_span, y0, h)

    factor = 1 - lam * h

    if 0 < factor < 1:
        behaviour = "Smooth decay"
    elif -1 < factor < 0:
        behaviour = "Oscillating decay"
    elif factor < -1:
        behaviour = "Growing oscillations"
    else:
        behaviour = "Boundary case"

    exact = y0 * np.exp(-lam * t)

    plt.plot(t, y, "o-", markersize=3, label=f"Euler h={h}")

    summary.append([
        h,
        lam * h,
        factor,
        behaviour,
        y[-1],
        exact[-1],
        abs(y[-1] - exact[-1])
    ])

    for i in range(len(t)):
        rows.append([
            h,
            t[i],
            y[i],
            exact[i],
            abs(y[i] - exact[i]),
            behaviour
        ])

plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Experiment 3: Euler Stability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "exp3_stability.png", dpi=200)
plt.close()


# Detailed results
with open(RESULTS / "exp3_stability.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "step_size", "time", "Euler_value",
        "exact_value", "absolute_error", "behaviour"
    ])
    writer.writerows(rows)


# Summary table
with open(
    RESULTS / "exp3_stability_summary.csv", "w", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "step_size", "lambda_h", "amplification_factor",
        "behaviour", "final_euler", "final_exact",
        "final_absolute_error"
    ])
    writer.writerows(summary)


print("Experiment 3 completed.")

for row in summary:
    print(
        f"h={row[0]:.2f}, lambda*h={row[1]:.2f}, "
        f"factor={row[2]:.2f}: {row[3]}"
)
