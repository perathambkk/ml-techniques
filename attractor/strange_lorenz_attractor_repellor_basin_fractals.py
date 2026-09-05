```python
"""
======================================================================
LORenz STRANGE REPELLOR LABORATORY
======================================================================

A deeper tutorial on attractors, repellors, invariant sets,
stable/unstable directions, Poincare sections, return maps,
Lyapunov exponents, and fractal dimension.

Lorenz equations:

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = xy - beta z

Parameters:

    sigma = 10
    rho   = 28
    beta  = 8/3

Forward time:
    strange attractor

Reverse time:
    strange repellor

The same invariant geometric set is traversed in the opposite
temporal direction.

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

# generated from ChatGPT using "generate me a tutorial code for a repellor" -> "Nice, go ahead."-> "Nice, go ahead."

# ======================================================================
# 1. PARAMETERS
# ======================================================================

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

DT = 0.01

TRANSIENT = 20.0
TOTAL_TIME = 100.0


# ======================================================================
# 2. LORENZ VECTOR FIELD
# ======================================================================

def lorenz(t, X):

    x, y, z = X

    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z

    return np.array([dx, dy, dz])


def lorenz_reverse(t, X):

    return -lorenz(t, X)


# ======================================================================
# 3. EQUILIBRIA
# ======================================================================

"""
The Lorenz system has three equilibria for rho > 1:

    E0 = (0,0,0)

and

    E+ = (+sqrt(beta(rho-1)),
          +sqrt(beta(rho-1)),
          rho-1)

    E- = (-sqrt(beta(rho-1)),
          -sqrt(beta(rho-1)),
          rho-1)
"""

q = np.sqrt(BETA * (RHO - 1))

E0 = np.array([0.0, 0.0, 0.0])
Eplus = np.array([q, q, RHO - 1])
Eminus = np.array([-q, -q, RHO - 1])

print("\nLorenz equilibria")
print("=================")
print("E0    =", E0)
print("Eplus =", Eplus)
print("Eminus=", Eminus)


# ======================================================================
# 4. GENERATE THE STRANGE ATTRACTOR
# ======================================================================

print("\nGenerating Lorenz invariant set...")

t = np.arange(
    0,
    TOTAL_TIME,
    DT
)

solution = solve_ivp(
    lorenz,
    [0, TOTAL_TIME],
    [1.0, 1.0, 1.0],
    t_eval=t,
    rtol=1e-9,
    atol=1e-11
)

X = solution.y.T

# Remove transient
keep = t >= TRANSIENT

X = X[keep:]
t = t[keep:]

x = X[:, 0]
y = X[:, 1]
z = X[:, 2]


# ======================================================================
# 5. 3-D STRANGE ATTRACTOR
# ======================================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

ax.plot(
    x,
    y,
    z,
    linewidth=0.4
)

ax.scatter(
    *Eplus,
    s=80,
    label="Equilibrium +"
)

ax.scatter(
    *Eminus,
    s=80,
    label="Equilibrium -"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title("Lorenz Strange Attractor")

ax.legend()

plt.show()


# ======================================================================
# 6. POINCARE SECTION
# ======================================================================

"""
Take the section

    z = rho - 1

and record upward crossings.
"""

SECTION_Z = RHO - 1

section_points = []

for i in range(len(X) - 1):

    z1 = X[i, 2]
    z2 = X[i + 1, 2]

    if z1 < SECTION_Z and z2 >= SECTION_Z:

        alpha = (SECTION_Z - z1) / (z2 - z1)

        point = X[i] + alpha * (X[i + 1] - X[i])

        section_points.append(point)

section_points = np.array(section_points)


plt.figure(figsize=(8, 7))

plt.scatter(
    section_points[:, 0],
    section_points[:, 1],
    s=4
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    r"Lorenz Poincaré Section ($z=\rho-1$)"
)

plt.grid(alpha=0.3)

plt.show()


# ======================================================================
# 7. LOCAL MAXIMA
# ======================================================================

"""
Extract local maxima of z(t).
"""

maxima = []

maxima_times = []

for i in range(1, len(z) - 1):

    if z[i] > z[i - 1] and z[i] > z[i + 1]:

        maxima.append(z[i])
        maxima_times.append(t[i])

maxima = np.array(maxima)
maxima_times = np.array(maxima_times)


plt.figure(figsize=(10, 5))

plt.plot(
    maxima_times,
    maxima,
    ".",
    markersize=3
)

plt.xlabel("time")
plt.ylabel("local maximum of z")

plt.title("Lorenz Local-Maximum Sequence")

plt.grid(alpha=0.3)

plt.show()


# ======================================================================
# 8. RETURN MAP
# ======================================================================

z_n = maxima[:-1]
z_next = maxima[1:]

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


# ======================================================================
# 9. BUILD A POINT-CLOUD APPROXIMATION OF THE INVARIANT SET
# ======================================================================

"""
The numerically generated trajectory approximates the invariant
Lorenz set.

We use this point cloud later to estimate:

    distance to invariant set

and

    escape/repulsion rates.
"""

tree = cKDTree(X)


# ======================================================================
# 10. PERTURB A POINT ON THE INVARIANT SET
# ======================================================================

"""
Pick a point from the attractor and perturb it very slightly.

The perturbation is transverse to the numerically generated
trajectory.

Under reverse-time dynamics the perturbation grows.

This is the key experiment.
"""

rng = np.random.default_rng(12345)

index = rng.integers(
    0,
    len(X)
)

X0 = X[index].copy()

direction = rng.normal(size=3)

direction /= np.linalg.norm(direction)

epsilon = 1e-6

X0_perturbed = X0 + epsilon * direction

print("\nRepellor experiment")
print("===================")

print("Point on invariant set:")
print(X0)

print("\nPerturbed point:")
print(X0_perturbed)

print("\nInitial perturbation:")
print(np.linalg.norm(X0_perturbed - X0))


# ======================================================================
# 11. FORWARD-TIME TRAJECTORY
# ======================================================================

"""
Start from the perturbed point and integrate forward.

This is NOT expected to show simple exponential repulsion from the
Lorenz attractor, because the Lorenz attractor is attracting in
forward time.
"""

forward_solution = solve_ivp(
    lorenz,
    [0, 30],
    X0_perturbed,
    t_eval=np.arange(0, 30, DT),
    rtol=1e-9,
    atol=1e-11
)

Xf = forward_solution.y.T


# ======================================================================
# 12. REVERSE-TIME TRAJECTORY
# ======================================================================

reverse_solution = solve_ivp(
    lorenz_reverse,
    [0, 30],
    X0_perturbed,
    t_eval=np.arange(0, 30, DT),
    rtol=1e-9,
    atol=1e-11
)

Xr = reverse_solution.y.T


# ======================================================================
# 13. COMPARE FORWARD AND REVERSE TRAJECTORIES
# ======================================================================

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection="3d")

ax1.plot(
    x,
    y,
    z,
    linewidth=0.3,
    alpha=0.3
)

ax1.plot(
    Xf[:, 0],
    Xf[:, 1],
    Xf[:, 2],
    linewidth=1
)

ax1.set_title(
    "Forward Time\nAttraction toward invariant set"
)

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")


ax2 = fig.add_subplot(122, projection="3d")

ax2.plot(
    x,
    y,
    z,
    linewidth=0.3,
    alpha=0.3
)

ax2.plot(
    Xr[:, 0],
    Xr[:, 1],
    Xr[:, 2],
    linewidth=1
)

ax2.set_title(
    "Reverse Time\nRepulsion from invariant set"
)

ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

plt.tight_layout()

plt.show()


# ======================================================================
# 14. DISTANCE TO INVARIANT SET
# ======================================================================

"""
For every point on the perturbed trajectory calculate

    d(t) = min ||X(t) - X_attractor||

using the KD-tree.

For the reverse-time trajectory, d(t) should initially grow
approximately exponentially.
"""

distance_forward, _ = tree.query(Xf)

distance_reverse, _ = tree.query(Xr)


plt.figure(figsize=(10, 6))

plt.semilogy(
    np.arange(len(distance_forward)) * DT,
    distance_forward,
    label="Forward time"
)

plt.semilogy(
    np.arange(len(distance_reverse)) * DT,
    distance_reverse,
    label="Reverse time"
)

plt.xlabel("time")
plt.ylabel("distance to invariant set")

plt.title(
    "Attraction versus Repulsion"
)

plt.legend()

plt.grid(alpha=0.3)

plt.show()


# ======================================================================
# 15. LOCAL REPULSION RATE
# ======================================================================

"""
If

    d(t) ≈ d0 exp(lambda*t)

then

    log(d(t)) ≈ log(d0) + lambda*t.

We fit the early-time logarithmic growth.
"""

time_reverse = np.arange(
    len(distance_reverse)
) * DT

mask = (
    (distance_reverse > 1e-6)
    &
    (distance_reverse < 1e-1)
)

if np.sum(mask) > 20:

    slope, intercept = np.polyfit(
        time_reverse[mask],
        np.log(distance_reverse[mask]),
        1
    )

    print("\nLocal reverse-time repulsion rate")
    print("=================================")

    print(
        f"lambda ~= {slope:.6f}"
    )


    plt.figure(figsize=(9, 6))

    plt.plot(
        time_reverse[mask],
        np.log(distance_reverse[mask]),
        ".",
        markersize=4
    )

    fitted = (
        intercept
        + slope * time_reverse[mask]
    )

    plt.plot(
        time_reverse[mask],
        fitted,
        linewidth=2
    )

    plt.xlabel("time")
    plt.ylabel("log(distance)")

    plt.title(
        "Exponential Repulsion: log d(t)"
    )

    plt.grid(alpha=0.3)

    plt.show()


# ======================================================================
# 16. LYAPUNOV EXPONENTS
# ======================================================================

"""
We estimate the Lyapunov spectrum using the standard
Benettin/QR-style tangent-space algorithm.

The Jacobian is

        [ -sigma       sigma       0 ]
J  =    [ rho-z        -1         -x ]
        [ y             x       -beta]

For reverse time:

    J_reverse = -J

Therefore the Lyapunov spectrum reverses sign.
"""


def jacobian(X):

    x, y, z = X

    return np.array([
        [-SIGMA, SIGMA, 0.0],
        [RHO - z, -1.0, -x],
        [y, x, -BETA]
    ])


def lyapunov_spectrum(
    rhs,
    jacobian_function,
    initial,
    total_time=50,
    dt=0.01
):

    Xstate = np.array(
        initial,
        dtype=float
    )

    Q = np.eye(3)

    sums = np.zeros(3)

    steps = int(total_time / dt)

    for k in range(steps):

        # Integrate state
        sol = solve_ivp(
            rhs,
            [0, dt],
            Xstate,
            t_eval=[dt],
            rtol=1e-8,
            atol=1e-10
        )

        Xstate = sol.y[:, -1]

        # Approximate tangent evolution
        J = jacobian_function(Xstate)

        M = np.eye(3) + dt * J

        Q = M @ Q

        Q, R = np.linalg.qr(Q)

        diagonal = np.abs(
            np.diag(R)
        )

        diagonal[
            diagonal == 0
        ] = 1e-300

        sums += np.log(diagonal)

    return sums / total_time


lambda_forward = lyapunov_spectrum(
    lorenz,
    jacobian,
    [1, 1, 1],
    total_time=50,
    dt=DT
)

lambda_reverse = lyapunov_spectrum(
    lorenz_reverse,
    lambda X: -jacobian(X),
    [1, 1, 1],
    total_time=50,
    dt=DT
)

lambda_forward = np.sort(
    lambda_forward
)[::-1]

lambda_reverse = np.sort(
    lambda_reverse
)[::-1]


print("\nLyapunov spectrum")
print("=================")

print(
    "Forward:",
    lambda_forward
)

print(
    "Reverse:",
    lambda_reverse
)

print(
    "\nNotice the approximate sign reversal."
)


# ======================================================================
# 17. KAPLAN-YORKE DIMENSION
# ======================================================================

"""
For ordered Lyapunov exponents

    lambda1 >= lambda2 >= lambda3

find the largest j for which

    lambda1 + ... + lambdaj >= 0.

Then

    D_KY = j + (lambda1 + ... + lambdaj)
                 / |lambda_(j+1)|

For the Lorenz attractor this gives a dimension around 2.06
for the usual parameter set, subject to numerical estimation
error.

This is a dynamical estimate, not a direct geometric measurement.
"""

def kaplan_yorke(lambdas):

    lambdas = np.sort(
        lambdas
    )[::-1]

    cumulative = np.cumsum(lambdas)

    j = 0

    for i, value in enumerate(cumulative):

        if value >= 0:

            j = i + 1

    if j == len(lambdas):

        return float(len(lambdas))

    if j == 0:

        return 0.0

    return (
        j
        + cumulative[j - 1]
        / abs(lambdas[j])
    )


DKY_forward = kaplan_yorke(
    lambda_forward
)

DKY_reverse = kaplan_yorke(
    lambda_reverse
)

print("\nKaplan-Yorke dimensions")
print("=======================")

print(
    f"Forward : {DKY_forward:.5f}"
)

print(
    f"Reverse : {DKY_reverse:.5f}"
)


# ======================================================================
# 18. SIMPLE BOX-COUNTING DIMENSION
# ======================================================================

"""
We can also estimate a fractal dimension directly from the
point cloud.

Procedure:

    1. Divide the (x,z) projection into boxes.
    2. Count occupied boxes.
    3. Repeat for several scales.
    4. Fit

         log N(epsilon)
         ----------------
         log(1/epsilon)

The resulting number is a rough projection dimension.
"""

projection = X[:, [0, 2]]

xmin, xmax = projection[:, 0].min(), projection[:, 0].max()
zmin, zmax = projection[:, 1].min(), projection[:, 1].max()

grid_sizes = np.array([
    8, 12, 16, 24, 32, 48, 64, 96
])

counts = []

for n in grid_sizes:

    ix = (
        (projection[:, 0] - xmin)
        / (xmax - xmin)
        * n
    ).astype(int)

    iz = (
        (projection[:, 1] - zmin)
        / (zmax - zmin)
        * n
    ).astype(int)

    ix = np.clip(ix, 0, n - 1)
    iz = np.clip(iz, 0, n - 1)

    occupied = set(
        zip(ix, iz)
    )

    counts.append(
        len(occupied)
    )

counts = np.array(counts)

epsilon = 1.0 / grid_sizes

fit = np.polyfit(
    np.log(1 / epsilon),
    np.log(counts),
    1
)

box_dimension = fit[0]

print("\nBox-counting estimate")
print("=====================")

print(
    f"Projection dimension ~= {box_dimension:.4f}"
)


plt.figure(figsize=(8, 6))

plt.plot(
    np.log(1 / epsilon),
    np.log(counts),
    "o"
)

plt.plot(
    np.log(1 / epsilon),
    np.polyval(
        fit,
        np.log(1 / epsilon)
    ),
    linewidth=2
)

plt.xlabel(r"$\log(1/\epsilon)$")
plt.ylabel(r"$\log N(\epsilon)$")

plt.title(
    "Box-Counting Scaling"
)

plt.grid(alpha=0.3)

plt.show()


# ======================================================================
# 19. STABLE/UNSTABLE LINEAR DIRECTIONS AT EQUILIBRIA
# ======================================================================

"""
The eigenvalues of the Jacobian at an equilibrium reveal local
stable and unstable directions.

This is the local building block from which global invariant
manifolds develop.
"""

for name, equilibrium in [
    ("E0", E0),
    ("E+", Eplus),
    ("E-", Eminus)
]:

    J = jacobian(equilibrium)

    eigenvalues, eigenvectors = np.linalg.eig(J)

    print(
        f"\n{name} eigenvalues:"
    )

    for value in eigenvalues:

        print(
            f"    {value.real:.6f}"
            f" {value.imag:+.6f}i"
        )


# ======================================================================
# 20. MANIFOLD-LIKE TRAJECTORIES
# ======================================================================

"""
We perturb the nonzero equilibrium E+ along its eigenvectors.

Forward integration follows its unstable directions.

Reverse integration follows the opposite temporal directions.

This provides an intuitive connection between:

    eigenvectors
        ->
    local invariant manifolds
        ->
    global strange geometry
"""

Jplus = jacobian(Eplus)

eigenvalues, eigenvectors = np.linalg.eig(Jplus)

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

ax.plot(
    x,
    y,
    z,
    linewidth=0.3,
    alpha=0.25
)

for k in range(3):

    direction = np.real(
        eigenvectors[:, k]
    )

    direction /= np.linalg.norm(direction)

    for sign in [-1, 1]:

        start = (
            Eplus
            + sign * 1e-4 * direction
        )

        sol = solve_ivp(
            lorenz,
            [0, 15],
            start,
            t_eval=np.arange(
                0,
                15,
                DT
            ),
            rtol=1e-8,
            atol=1e-10
        )

        M = sol.y.T

        ax.plot(
            M[:, 0],
            M[:, 1],
            M[:, 2],
            linewidth=1
        )

ax.scatter(
    *Eplus,
    s=100
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title(
    "Local Eigen-Directions and Global Lorenz Geometry"
)

plt.show()


# ======================================================================
# 21. REPELLOR ANIMATION
# ======================================================================

"""
Animation:

    faint curve = invariant Lorenz set
    moving point = reverse-time trajectory
    tail = trajectory history

The point begins extremely close to the invariant set and
moves away under reverse-time dynamics.
"""

trajectory = Xr

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

# Invariant set
ax.plot(
    x,
    y,
    z,
    linewidth=0.3,
    alpha=0.25
)

tail, = ax.plot(
    [],
    [],
    [],
    linewidth=1.5
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
    Xr[:, 0].min() - 5,
    Xr[:, 0].max() + 5
)

ax.set_ylim(
    Xr[:, 1].min() - 5,
    Xr[:, 1].max() + 5
)

ax.set_zlim(
    Xr[:, 2].min() - 5,
    Xr[:, 2].max() + 5
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title(
    "Strange Repellor: Reverse-Time Escape"
)


def init():

    tail.set_data([], [])
    tail.set_3d_properties([])

    point.set_data([], [])
    point.set_3d_properties([])

    return tail, point


def update(frame):

    history = trajectory[
        max(0, frame - 1000):frame
    ]

    tail.set_data(
        history[:, 0],
        history[:, 1]
    )

    tail.set_3d_properties(
        history[:, 2]
    )

    point.set_data(
        [trajectory[frame, 0]],
        [trajectory[frame, 1]]
    )

    point.set_3d_properties(
        [trajectory[frame, 2]]
    )

    return tail, point


anim = FuncAnimation(
    fig,
    update,
    frames=len(trajectory),
    init_func=init,
    interval=20,
    blit=False
)

plt.show()


# ======================================================================
# 22. FINAL SUMMARY
# ======================================================================

print()
print("=" * 70)
print("CONCEPTUAL SUMMARY")
print("=" * 70)

print("""
1. The Lorenz system possesses a complicated invariant set.

2. In forward time, nearby trajectories are attracted toward
   this set.

3. Reverse the vector field:

       F(X)  ->  -F(X)

   and the temporal stability reverses.

4. The same geometric invariant set becomes a repellor.

5. A trajectory started exactly on the invariant set remains
   on the set.

6. A trajectory started infinitesimally away from it can move
   exponentially away under the reversed dynamics.

7. The Poincare section converts the continuous-time flow into
   a lower-dimensional return map.

8. The return map exposes the nonlinear structure hidden inside
   the three-dimensional flow.

9. Positive and negative Lyapunov exponents exchange roles
   under time reversal.

10. The complicated geometry is not caused by the repellor
    being a single unstable equilibrium.

    The repellor is a complicated invariant set.

Therefore:

              ATTRACTOR
                  |
                  | reverse time
                  v
              REPELLOR

and

       stable directions <--> unstable directions

under time reversal.
""")

print("=" * 70)
print("Laboratory complete.")
print("=" * 70)
```
