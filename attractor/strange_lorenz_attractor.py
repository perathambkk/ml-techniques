"""
Tutorial: The Lorenz Strange Attractor
======================================
# generated from ChatGPT using "generate me a tutorial code for a strange attractor"

The Lorenz system is a classic example of deterministic chaos.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

For the classic parameter choice

    sigma = 10
    rho   = 28
    beta  = 8/3

the trajectory does not settle to a fixed point or a simple periodic orbit.
Instead, it approaches a complicated geometric object called a
STRANGE ATTRACTOR.

Requirements:
    numpy
    matplotlib

Install:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Lorenz equations
# ============================================================

def lorenz(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Compute the velocity vector (dx/dt, dy/dt, dz/dt)
    for the Lorenz dynamical system.
    """

    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.array([dx, dy, dz])


# ============================================================
# 2. Numerical integration
# ============================================================

def rk4_step(state, dt):
    """
    One fourth-order Runge-Kutta (RK4) integration step.
    """

    k1 = lorenz(state)
    k2 = lorenz(state + 0.5 * dt * k1)
    k3 = lorenz(state + 0.5 * dt * k2)
    k4 = lorenz(state + dt * k3)

    return state + (dt / 6.0) * (
        k1 + 2*k2 + 2*k3 + k4
    )


def generate_trajectory(
    initial_state,
    dt=0.01,
    steps=100_000
):
    """
    Generate a Lorenz trajectory using RK4.
    """

    trajectory = np.zeros((steps + 1, 3))
    trajectory[0] = initial_state

    state = np.array(initial_state, dtype=float)

    for i in range(steps):
        state = rk4_step(state, dt)
        trajectory[i + 1] = state

    return trajectory


# ============================================================
# 3. Generate the attractor
# ============================================================

initial_state = np.array([1.0, 1.0, 1.0])

trajectory = generate_trajectory(
    initial_state,
    dt=0.01,
    steps=100_000
)

x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]


# ============================================================
# 4. Remove the transient
# ============================================================
#
# The initial part of the trajectory depends strongly on the
# starting point. After some time, the orbit approaches the
# attractor.
#

transient = 5_000

x_a = x[transient:]
y_a = y[transient:]
z_a = z[transient:]


# ============================================================
# 5. Print some information
# ============================================================

print("=" * 60)
print("LORENZ STRANGE ATTRACTOR")
print("=" * 60)

print("\nParameters:")
print("sigma =", 10.0)
print("rho   =", 28.0)
print("beta  =", 8.0 / 3.0)

print("\nInitial state:")
print(initial_state)

print("\nNumber of integration steps:")
print(len(trajectory) - 1)

print("\nTime step:")
print(0.01)

print("\nAttractor statistics after transient removal:")

print("x range:",
      np.min(x_a),
      "to",
      np.max(x_a))

print("y range:",
      np.min(y_a),
      "to",
      np.max(y_a))

print("z range:",
      np.min(z_a),
      "to",
      np.max(z_a))

print("\nMean state:")
print(np.mean(trajectory[transient:], axis=0))

print("\nStandard deviation:")
print(np.std(trajectory[transient:], axis=0))


# ============================================================
# 6. Show several trajectory points
# ============================================================

print("\nFirst 10 points on the attractor:")

for i in range(10):
    print(
        f"{i:3d}: "
        f"x={x_a[i]: .6f}, "
        f"y={y_a[i]: .6f}, "
        f"z={z_a[i]: .6f}"
    )


# ============================================================
# 7. Plot the 3D strange attractor
# ============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

ax.plot(
    x_a,
    y_a,
    z_a,
    linewidth=0.4
)

ax.set_title("Lorenz Strange Attractor")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.tight_layout()
plt.show()


# ============================================================
# 8. Plot the x-y projection
# ============================================================

plt.figure(figsize=(9, 7))

plt.plot(
    x_a,
    y_a,
    linewidth=0.3
)

plt.title("Lorenz Attractor — x-y Projection")

plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()


# ============================================================
# 9. Plot the x-z projection
# ============================================================

plt.figure(figsize=(9, 7))

plt.plot(
    x_a,
    z_a,
    linewidth=0.3
)

plt.title("Lorenz Attractor — x-z Projection")

plt.xlabel("x")
plt.ylabel("z")

plt.tight_layout()
plt.show()

# ============================================================
# 10. Sensitivity to initial conditions
# ============================================================

state1 = np.array([1.0, 1.0, 1.0])

# Only change x by an extremely tiny amount
state2 = np.array([1.0 + 1e-10, 1.0, 1.0])

steps = 10_000
dt = 0.01

distances = np.zeros(steps + 1)

for i in range(steps + 1):

    distances[i] = np.linalg.norm(state1 - state2)

    state1 = rk4_step(state1, dt)
    state2 = rk4_step(state2, dt)


# Plot separation of the two trajectories

plt.figure(figsize=(9, 6))

plt.semilogy(
    np.arange(steps + 1) * dt,
    distances
)

plt.xlabel("Time")
plt.ylabel("Distance between trajectories")

plt.title(
    "Sensitive Dependence on Initial Conditions"
)

plt.tight_layout()
plt.show()