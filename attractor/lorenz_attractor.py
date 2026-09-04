import numpy as np
import matplotlib.pyplot as plt

# generated from ChatGPT using "Generate me a tutorial code for an attractor"

# --------------------------------------------------
# 1. Define the dynamical system
# --------------------------------------------------

def lorenz(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.array([dx, dy, dz])


# --------------------------------------------------
# 2. Simple RK4 numerical integrator
# --------------------------------------------------

def rk4_step(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)

    return state + (dt / 6.0) * (
        k1 + 2*k2 + 2*k3 + k4
    )


# --------------------------------------------------
# 3. Generate the trajectory
# --------------------------------------------------

dt = 0.01
steps = 100_000

state = np.array([1.0, 1.0, 1.0])

trajectory = np.empty((steps, 3))

for i in range(steps):
    trajectory[i] = state
    state = rk4_step(lorenz, state, dt)


# --------------------------------------------------
# 4. Plot the attractor
# --------------------------------------------------

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

ax.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    trajectory[:, 2],
    linewidth=0.5
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Lorenz Attractor")

plt.show()