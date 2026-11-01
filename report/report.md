Experiment 3: Stability of Euler's Method

We studied Euler's method for the differential equation

[
y'=-\lambda y,\qquad \lambda>0.
]

The amplification factor is (1-\lambda h). The numerical solution is stable when (|1-\lambda h|<1), which gives (0<\lambda h<2).

For (\lambda=10):

- (h=0.05): amplification factor (0.5); the solution decays smoothly.
- (h=0.15): amplification factor (-0.5); the solution oscillates while decaying.
- (h=0.25): amplification factor (-1.5); the solution oscillates with increasing magnitude and is unstable.

Experiment 4: Harmonic Oscillator

We considered the system

[
y_1'=y_2,\qquad y_2'=-y_1,
]

with initial conditions (y_1(0)=1) and (y_2(0)=0). The exact solution follows a circle of radius 1 in the phase plane.

The final radius obtained using Euler's method was approximately 1.366, while RK4 gave approximately 0.99999957. The exact radius is 1.0.

Euler's method causes the numerical trajectory to spiral outward, whereas RK4 preserves the circular trajectory much more accurately for the chosen step size.

The plots and numerical results are saved in the "results" directory.
