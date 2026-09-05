```python
"""
===============================================================
Lorenz Strange Repellor — Tutorial
===============================================================

We start with the classical Lorenz equations

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x*y - beta*z

with

    sigma = 10
    rho   = 28
    beta  = 8/3

For forward time, the Lorenz system has a famous strange
attractor.

If we reverse time,

    dx/dtau = -dx/dt
    dy/dtau = -dy/dt
    dz/dtau = -dz/dt

the same invariant set becomes a STRANGE REPELLOR.

This tutorial demonstrates:

    1. Lorenz forward attractor
    2. Reverse-time repellor
    3. 3-D phase-space visualization
    4. Poincare section
    5. Local maxima
    6. Return map
    7. Distance from the invariant set
    8. Lyapunov exponent estimation
    9. Animated repulsion
   10. Numerical observations

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

# generated from ChatGPT using "generate me a tutorial code for a repellor" -> "Nice, go ahead."

# =============================================================
# 1. Lorenz parameters
# =============================================================

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


# =============================================================
# 2. Lorenz equations
# =============================================================

def lorenz(t, state):
    """
    Classical Lorenz system.
    """

    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.array([dx, dy, dz])


# =============================================================
# 3. Reverse-time Lorenz system
# =============================================================

def lorenz_reverse(t, state):
    """
    Reverse-time Lorenz dynamics.

    This is simply

        dX/dtau = -F(X)

    where F is the ordinary Lorenz vector field.
    """

    return -lorenz(t, state)


# =============================================================
# 4. Integrate the ordinary Lorenz attractor
# =============================================================

print("Generating Lorenz attractor...")

initial_state = [1.0, 1.0, 1.0]

T = 60.0
dt = 0.01

t_eval = np.arange(0.0, T, dt)

solution_forward = solve_ivp(
    lorenz,
    [0.0, T],
    initial_state,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

X_forward = solution_forward.y.T

x = X_forward[:, 0]
y = X_forward[:, 1]
z = X_forward[:, 2]


# =============================================================
# 5. Remove transient
# =============================================================

transient = 1000

X_forward = X_forward[transient:]

x = X_forward[:, 0]
y = X_forward[:, 1]
z = X_forward[:, 2]


# =============================================================
# 6. Plot ordinary Lorenz attractor
# =============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

ax.plot(
    x,
    y,
    z,
    linewidth=0.5
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title("Lorenz Strange Attractor")

plt.show()


# =============================================================
# 7. Generate reverse-time trajectories
# =============================================================

print("Generating reverse-time trajectories...")

# Several points near the invariant set
# are perturbed slightly.

rng = np.random.default_rng(42)

num_trajectories = 8

reverse_trajectories = []

for i in range(num_trajectories):

    # Choose a point from the Lorenz attractor
    index = rng.integers(0, len(X_forward))

    base = X_forward[index]

    # Small perturbation
    perturbation = 1e-3 * rng.normal(size=3)

    initial = base + perturbation

    solution = solve_ivp(
        lorenz_reverse,
        [0.0, 12.0],
        initial,
        t_eval=np.arange(0.0, 12.0, 0.01),
        rtol=1e-8,
        atol=1e-10
    )

    reverse_trajectories.append(solution.y.T)


# =============================================================
# 8. Plot the strange repellor
# =============================================================

fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(111, projection="3d")

# The invariant set itself
ax.plot(
    X_forward[:, 0],
    X_forward[:, 1],
    X_forward[:, 2],
    linewidth=0.4,
    alpha=0.25,
    label="Lorenz invariant set"
)

# Reverse-time trajectories
for i, trajectory in enumerate(reverse_trajectories):

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        linewidth=1.2,
        label="repelled trajectory" if i == 0 else None
    )

    # Starting point
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        s=30
    )


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title(
    "Lorenz Strange Repellor\n"
    "Reverse-Time Trajectories Move Away from the Invariant Set"
)

ax.legend()

plt.show()


# =============================================================
# 9. Poincare section
# =============================================================

"""
A common Poincare section is

    z = rho - 1

We record crossings where

    z[n] < rho-1
    z[n+1] >= rho-1

This converts the continuous-time system into a
lower-dimensional discrete dynamical system.
"""

section_z = rho - 1.0

crossings = []

for i in range(len(X_forward) - 1):

    z1 = X_forward[i, 2]
    z2 = X_forward[i + 1, 2]

    if z1 < section_z and z2 >= section_z:

        # Linear interpolation
        alpha = (section_z - z1) / (z2 - z1)

        point = (
            X_forward[i]
            + alpha * (X_forward[i + 1] - X_forward[i])
        )

        crossings.append(point)


crossings = np.array(crossings)


plt.figure(figsize=(8, 7))

plt.scatter(
    crossings[:, 0],
    crossings[:, 1],
    s=5
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    r"Poincaré Section: $z = \rho - 1$"
)

plt.grid(alpha=0.3)

plt.show()


# =============================================================
# 10. Local maxima of z
# =============================================================

"""
A useful scalar observable is the sequence of local maxima

    z_1, z_2, z_3, ...

of the Lorenz z(t) coordinate.
"""

local_maxima = []

for i in range(1, len(z) - 1):

    if z[i] > z[i - 1] and z[i] > z[i + 1]:

        local_maxima.append(z[i])


local_maxima = np.array(local_maxima)


plt.figure(figsize=(10, 5))

plt.plot(
    local_maxima,
    ".",
    markersize=3
)

plt.xlabel("Peak number")
plt.ylabel("z maximum")

plt.title("Lorenz Local-Maximum Sequence")

plt.grid(alpha=0.3)

plt.show()


# =============================================================
# 11. Return map
# =============================================================

"""
The return map plots

    z_(n+1)

against

    z_n.

For a periodic orbit this would collapse to a finite set.

For chaotic dynamics it produces a complicated curve.
"""

z_n = local_maxima[:-1]
z_next = local_maxima[1:]


plt.figure(figsize=(8, 7))

plt.scatter(
    z_n,
    z_next,
    s=5
)

plt.xlabel(r"$z_n$")
plt.ylabel(r"$z_{n+1}$")

plt.title("Lorenz Return Map")

plt.grid(alpha=0.3)

plt.show()


# =============================================================
# 12. Estimate Lyapunov exponent
# =============================================================

"""
For a continuous dynamical system, the largest Lyapunov
exponent measures exponential separation:

    ||delta X(t)|| ~ exp(lambda*t)

For the ordinary Lorenz attractor:

    lambda_max > 0

For the reversed system, the Lyapunov exponents change sign.

Thus the strange attractor's positive exponent becomes
negative under time reversal.

However, the invariant set is still repelling in the
appropriate dynamical directions because the complete
spectrum is reversed.
"""


def estimate_largest_lyapunov(
    rhs,
    initial,
    dt=0.01,
    steps=50000,
    delta0=1e-8
):
    """
    Simple two-trajectory Lyapunov estimate.

    Repeatedly evolve two nearby trajectories, measure their
    separation, renormalize, and accumulate logarithmic growth.
    """

    x1 = np.array(initial, dtype=float)

    # Random perturbation
    direction = np.array([1.0, 1.0, 1.0])
    direction /= np.linalg.norm(direction)

    x2 = x1 + delta0 * direction

    total = 0.0
    count = 0

    for k in range(steps):

        sol1 = solve_ivp(
            rhs,
            [0, dt],
            x1,
            t_eval=[dt],
            rtol=1e-8,
            atol=1e-10
        )

        sol2 = solve_ivp(
            rhs,
            [0, dt],
            x2,
            t_eval=[dt],
            rtol=1e-8,
            atol=1e-10
        )

        x1 = sol1.y[:, -1]
        x2 = sol2.y[:, -1]

        separation = np.linalg.norm(x2 - x1)

        if separation == 0:
            continue

        total += np.log(separation / delta0)

        # Renormalize
        x2 = x1 + delta0 * (x2 - x1) / separation

        count += 1

    return total / (count * dt)


print("\nLyapunov analysis")
print("=================")

lambda_forward = estimate_largest_lyapunov(
    lorenz,
    [1.0, 1.0, 1.0],
    dt=0.01,
    steps=5000
)

lambda_reverse = estimate_largest_lyapunov(
    lorenz_reverse,
    [1.0, 1.0, 1.0],
    dt=0.01,
    steps=5000
)

print(
    f"Forward-time estimate : {lambda_forward:.5f}"
)

print(
    f"Reverse-time estimate : {lambda_reverse:.5f}"
)

print(
    "\nThe exponents change sign under time reversal "
    "in the ideal mathematical system."
)


# =============================================================
# 13. Distance from the invariant set
# =============================================================

"""
To make "repulsion" quantitative, approximate the Lorenz
invariant set by the point cloud X_forward.

For each reverse-time trajectory we measure

    d(t) = distance to the nearest point in the invariant set.

If the trajectory is being repelled, d(t) tends to increase.
"""

tree = cKDTree(X_forward)


plt.figure(figsize=(10, 6))

for trajectory in reverse_trajectories:

    distances, indices = tree.query(
        trajectory
    )

    # Ignore very late times if the trajectory has escaped
    distances = np.minimum(distances, 100)

    plt.semilogy(
        np.arange(len(distances)) * 0.01,
        distances,
        linewidth=1
    )


plt.xlabel("Reverse time")
plt.ylabel("Distance to invariant set")

plt.title(
    "Trajectories Being Repelled from the Lorenz Invariant Set"
)

plt.grid(alpha=0.3)

plt.show()


# =============================================================
# 14. Estimate exponential departure
# =============================================================

"""
At sufficiently small distances we expect approximately

    d(t) ~ d(0) exp(lambda*t)

so

    log d(t) ~ lambda*t.
"""

trajectory = reverse_trajectories[0]

distances, indices = tree.query(trajectory)

times = np.arange(len(distances)) * 0.01

# Only fit the region where the trajectory remains relatively
# close to the invariant set.
mask = (
    (distances > 1e-4)
    &
    (distances < 1e-1)
)

if np.sum(mask) > 10:

    slope, intercept = np.polyfit(
        times[mask],
        np.log(distances[mask]),
        1
    )

    print(
        "\nLocal repulsion-rate estimate:"
    )

    print(
        f"slope = {slope:.5f}"
    )


# =============================================================
# 15. Animated reverse-time trajectory
# =============================================================

"""
The animation shows a point initially close to the strange
invariant set and then follows it under reverse-time dynamics.

This gives an intuitive visual interpretation of a repellor.
"""

animation_trajectory = reverse_trajectories[0]

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

# Plot invariant set faintly
ax.plot(
    X_forward[:, 0],
    X_forward[:, 1],
    X_forward[:, 2],
    linewidth=0.3,
    alpha=0.25
)

line, = ax.plot(
    [],
    [],
    [],
    linewidth=2
)

point, = ax.plot(
    [],
    [],
    [],
    marker="o",
    markersize=7,
    linestyle=""
)


ax.set_xlim(
    np.min(X_forward[:, 0]) - 10,
    np.max(X_forward[:, 0]) + 10
)

ax.set_ylim(
    np.min(X_forward[:, 1]) - 10,
    np.max(X_forward[:, 1]) + 10
)

ax.set_zlim(
    np.min(X_forward[:, 2]) - 10,
    np.max(X_forward[:, 2]) + 10
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title(
    "Reverse-Time Lorenz Dynamics: Repulsion"
)


def init():

    line.set_data([], [])
    line.set_3d_properties([])

    point.set_data([], [])
    point.set_3d_properties([])

    return line, point


def update(frame):

    trajectory = animation_trajectory[:frame]

    line.set_data(
        trajectory[:, 0],
        trajectory[:, 1]
    )

    line.set_3d_properties(
        trajectory[:, 2]
    )

    point.set_data(
        [animation_trajectory[frame, 0]],
        [animation_trajectory[frame, 1]]
    )

    point.set_3d_properties(
        [animation_trajectory[frame, 2]]
    )

    return line, point


anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=len(animation_trajectory),
    interval=20,
    blit=False
)

plt.show()


# =============================================================
# 16. Summary
# =============================================================

print()
print("===================================================")
print("SUMMARY")
print("===================================================")

print("""
The classical Lorenz system has a strange attractor.

Under time reversal:

    dX/dt = F(X)

becomes

    dX/dtau = -F(X).

The invariant set itself is unchanged as a set of points,
but its dynamical stability is reversed.

Therefore:

    attractor  <-->  repellor

under time reversal.

The important distinction is:

    An attractor pulls nearby trajectories toward it.

    A repellor pushes nearby trajectories away from it.

The geometry can remain extremely complicated.

Thus a repellor need not be a single unstable fixed point.

It can instead be a complicated fractal invariant set.
""")

print("Done.")
```
