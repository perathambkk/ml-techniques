# NLP Attractor–Repellor Dynamical-System Laboratory
# generated from ChatGPT using "generate me a tutorial code for an attractor-repellor for NLP (words)" -> "Nice, all of them, go ahead!"

## 0. Installation

```bash
pip install numpy scipy matplotlib scikit-learn sentence-transformers
```

Then run the program.

---

# 1. Complete program

```python
"""
===============================================================
 NLP ATTRACTOR–REPELLOR DYNAMICAL SYSTEM LABORATORY
===============================================================

Features
--------
1. Real transformer embeddings
2. Semantic attractors
3. Semantic repellors
4. Pairwise word forces
5. PCA visualization
6. Semantic basins of attraction
7. Phase portraits
8. Word trajectories
9. Poincare sections
10. Local maxima
11. Return maps
12. Lyapunov exponent estimation
13. Convergence/divergence diagnostics
14. Animated trajectories

The embedding space is high-dimensional.

For visualization we project the embedding space to 2-D with PCA.

The dynamical system itself is defined in the original
embedding space.

===============================================================
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer


# =============================================================
# 1. REPRODUCIBILITY
# =============================================================

SEED = 42

np.random.seed(SEED)


# =============================================================
# 2. WORD VOCABULARY
# =============================================================

categories = {

    "animal": [
        "cat",
        "dog",
        "lion",
        "tiger",
        "wolf",
        "horse"
    ],

    "vehicle": [
        "car",
        "bus",
        "train",
        "airplane",
        "ship",
        "motorcycle"
    ],

    "food": [
        "pizza",
        "bread",
        "rice",
        "apple",
        "banana",
        "cheese"
    ],

    "emotion": [
        "love",
        "happiness",
        "sadness",
        "anger",
        "fear",
        "joy"
    ]
}


words = [
    word
    for group in categories.values()
    for word in group
]


# =============================================================
# 3. LOAD REAL NLP EMBEDDING MODEL
# =============================================================

print("\nLoading transformer model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =============================================================
# 4. CREATE REAL WORD EMBEDDINGS
# =============================================================

print("Encoding words...")

raw_embeddings = model.encode(
    words,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(
    raw_embeddings,
    dtype=float
)


word_to_index = {
    word: i
    for i, word in enumerate(words)
}


# =============================================================
# 5. COMPUTE SEMANTIC ATTRACTORS
# =============================================================
#
# Each attractor is the centroid of a semantic category.
#
#                cat
#              /     \
#           dog       lion
#              \     /
#              ANIMAL
#
# =============================================================

attractors = {}

for category, group_words in categories.items():

    indices = [
        word_to_index[w]
        for w in group_words
    ]

    centroid = np.mean(
        embeddings[indices],
        axis=0
    )

    centroid /= np.linalg.norm(centroid)

    attractors[category] = centroid


# =============================================================
# 6. CONSTRUCT REPELLORS
# =============================================================
#
# A simple construction:
#
# repellor(category)
#
# is placed approximately opposite the corresponding
# attractor in embedding space.
#
# =============================================================

repellors = {}

for category, attractor in attractors.items():

    r = -attractor

    r /= np.linalg.norm(r)

    repellors[category] = r


# =============================================================
# 7. DYNAMICAL-SYSTEM PARAMETERS
# =============================================================

ATTRACTION = 1.25
REPULSION = 0.35

DAMPING = 0.08

INTERACTION = 0.20

REPULSION_EPS = 0.10

DT = 0.025

STEPS = 1000


# =============================================================
# 8. NORMALIZATION
# =============================================================

def safe_normalize(x):

    norm = np.linalg.norm(x)

    if norm < 1e-12:
        return x

    return x / norm


# =============================================================
# 9. VECTOR FIELD
# =============================================================

def vector_field(
    x,
    target_category,
    all_attractors=True
):

    """
    dx/dt =
    
        semantic attraction
        + semantic repulsion
        + weak nonlinear interaction
        - damping
    """

    attractor = attractors[target_category]
    repellor = repellors[target_category]

    # ---------------------------------------------------------
    # Attraction
    # ---------------------------------------------------------

    attraction_force = (
        attractor - x
    )

    # ---------------------------------------------------------
    # Repulsion
    # ---------------------------------------------------------

    difference = x - repellor

    distance = np.linalg.norm(
        difference
    )

    repulsion_force = (
        difference
        /
        (distance**2 + REPULSION_EPS)
    )

    # ---------------------------------------------------------
    # Nonlinear semantic interaction
    #
    # The tanh term makes the system nonlinear.
    # ---------------------------------------------------------

    nonlinear_force = np.tanh(x)

    # ---------------------------------------------------------
    # Total field
    # ---------------------------------------------------------

    velocity = (

        ATTRACTION
        * attraction_force

        +

        REPULSION
        * repulsion_force

        +

        INTERACTION
        * nonlinear_force

        -

        DAMPING
        * x
    )

    return velocity


# =============================================================
# 10. EULER INTEGRATOR
# =============================================================

def integrate(
    x0,
    target_category,
    steps=STEPS
):

    trajectory = np.zeros(
        (steps + 1, len(x0))
    )

    trajectory[0] = x0

    x = x0.copy()

    for t in range(steps):

        dx = vector_field(
            x,
            target_category
        )

        x = x + DT * dx

        # Keep state bounded on the unit sphere
        x = safe_normalize(x)

        trajectory[t + 1] = x

    return trajectory


# =============================================================
# 11. SIMULATE ALL WORDS
# =============================================================

trajectories = {}

print("\nSimulating word trajectories...")

for category, group_words in categories.items():

    for word in group_words:

        trajectory = integrate(
            embeddings[word_to_index[word]],
            category
        )

        trajectories[word] = trajectory


# =============================================================
# 12. PCA: HIGH-DIMENSIONAL → 2-D
# =============================================================
#
# IMPORTANT:
#
# PCA is ONLY used for visualization.
#
# The dynamics above occurred in the original
# transformer embedding space.
#
# =============================================================

all_points = []

for word in words:

    all_points.append(
        trajectories[word]
    )

for category in categories:

    all_points.append(
        attractors[category][None, :]
    )

all_points = np.vstack(
    all_points
)


pca = PCA(
    n_components=2
)

pca.fit(all_points)


def project(x):

    return pca.transform(x)


# =============================================================
# 13. PROJECT ATTRACTORS / REPELLORS
# =============================================================

projected_attractors = {
    category:
    project(attractors[category][None, :])[0]

    for category in attractors
}


projected_repellors = {
    category:
    project(repellors[category][None, :])[0]

    for category in repellors
}


# =============================================================
# 14. PROJECT WORD TRAJECTORIES
# =============================================================

projected_trajectories = {

    word:
    project(trajectory)

    for word, trajectory
    in trajectories.items()
}


# =============================================================
# 15. PHASE PORTRAIT
# =============================================================

plt.figure(figsize=(13, 10))

for word in words:

    trajectory = projected_trajectories[word]

    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        alpha=0.35,
        linewidth=1
    )

    plt.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=25
    )

    plt.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        s=50
    )

    plt.text(
        trajectory[-1, 0],
        trajectory[-1, 1],
        word,
        fontsize=9
    )


# Attractors

for category, point in projected_attractors.items():

    plt.scatter(
        point[0],
        point[1],
        marker="*",
        s=400,
        edgecolors="black",
        linewidths=1.5
    )

    plt.text(
        point[0],
        point[1],
        f"  ATTRACTOR\n  {category}",
        fontsize=10,
        fontweight="bold"
    )


# Repellors

for category, point in projected_repellors.items():

    plt.scatter(
        point[0],
        point[1],
        marker="X",
        s=250,
        edgecolors="black",
        linewidths=1.5
    )

    plt.text(
        point[0],
        point[1],
        f"  REPELLOR\n  {category}",
        fontsize=9
    )


plt.title(
    "NLP Word Attractor–Repellor Phase Portrait"
)

plt.xlabel("Principal semantic dimension 1")
plt.ylabel("Principal semantic dimension 2")

plt.grid(alpha=0.25)

plt.tight_layout()

plt.show()


# =============================================================
# 16. DISTANCE TO ATTRACTOR
# =============================================================

print("\n")
print("=" * 70)
print("FINAL DISTANCES TO SEMANTIC ATTRACTORS")
print("=" * 70)

for category, group_words in categories.items():

    attractor = attractors[category]

    print(f"\n[{category.upper()}]")

    for word in group_words:

        final_state = trajectories[word][-1]

        distance = np.linalg.norm(
            final_state - attractor
        )

        print(
            f"{word:12s} "
            f"distance = {distance:.6f}"
        )


# =============================================================
# 17. SEMANTIC CONVERGENCE RATE
# =============================================================

print("\n")
print("=" * 70)
print("CONVERGENCE RATES")
print("=" * 70)

for category, group_words in categories.items():

    attractor = attractors[category]

    print(f"\n[{category.upper()}]")

    for word in group_words:

        trajectory = trajectories[word]

        distances = np.linalg.norm(
            trajectory - attractor,
            axis=1
        )

        initial = distances[0]
        final = distances[-1]

        rate = (
            initial - final
        ) / max(initial, 1e-12)

        print(
            f"{word:12s} "
            f"initial={initial:.4f} "
            f"final={final:.4f} "
            f"reduction={rate:.4f}"
        )


# =============================================================
# 18. POINCARE SECTION
# =============================================================
#
# We define a section:
#
#        PCA-x = 0
#
# and record intersections when the trajectory crosses it
# in a specified direction.
#
# This converts continuous trajectories into a sequence of
# discrete observations.
#
# =============================================================

def poincare_section(
    trajectory_2d,
    axis=0,
    section_value=0.0
):

    values = (
        trajectory_2d[:, axis]
        - section_value
    )

    crossings = []

    for i in range(
        len(values) - 1
    ):

        if values[i] < 0 and values[i + 1] >= 0:

            crossings.append(
                trajectory_2d[i + 1]
            )

    return np.asarray(crossings)


poincare_data = {}

for word in words:

    section = poincare_section(
        projected_trajectories[word]
    )

    poincare_data[word] = section


# Plot Poincare section

plt.figure(figsize=(10, 8))

for word, section in poincare_data.items():

    if len(section) == 0:
        continue

    plt.scatter(
        np.arange(len(section)),
        section[:, 1],
        s=25,
        alpha=0.7
    )


plt.title(
    "Poincaré Section of Word Trajectories"
)

plt.xlabel(
    "Crossing number"
)

plt.ylabel(
    "Second semantic coordinate"
)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.show()


# =============================================================
# 19. LOCAL MAXIMA
# =============================================================
#
# We examine oscillations in semantic coordinate y(t).
#
# A local maximum satisfies:
#
#       y[i-1] < y[i] > y[i+1]
#
# =============================================================

def local_maxima(signal):

    maxima = []

    for i in range(
        1,
        len(signal) - 1
    ):

        if (
            signal[i] > signal[i - 1]
            and
            signal[i] > signal[i + 1]
        ):

            maxima.append(i)

    return np.asarray(maxima)


local_max_data = {}

for word in words:

    trajectory = projected_trajectories[word]

    y = trajectory[:, 1]

    indices = local_maxima(y)

    local_max_data[word] = (
        indices,
        y[indices]
    )


# =============================================================
# 20. DISPLAY LOCAL MAXIMA
# =============================================================

print("\n")
print("=" * 70)
print("LOCAL MAXIMA")
print("=" * 70)

for word in words:

    indices, values = local_max_data[word]

    print(
        f"\n{word}:"
    )

    print(
        "indices =",
        indices[:20]
    )

    print(
        "values  =",
        np.round(values[:20], 5)
    )


# =============================================================
# 21. RETURN MAP
# =============================================================
#
# If local maxima are:
#
#       M1, M2, M3, ...
#
# then construct:
#
#       (M1, M2)
#       (M2, M3)
#       (M3, M4)
#
# This is the return map.
#
# A fixed point corresponds approximately to:
#
#       M(n+1) = M(n)
#
# Periodic dynamics produce cycles.
#
# =============================================================

plt.figure(figsize=(10, 8))

for word in words:

    indices, maxima = local_max_data[word]

    if len(maxima) < 2:
        continue

    plt.scatter(
        maxima[:-1],
        maxima[1:],
        s=25,
        alpha=0.7
    )


plt.xlabel(
    "Local maximum M(n)"
)

plt.ylabel(
    "Local maximum M(n+1)"
)

plt.title(
    "Return Map of Semantic Word Dynamics"
)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.show()


# =============================================================
# 22. MORE USEFUL RETURN MAP
# =============================================================
#
# Instead of using sample indices, use actual semantic
# amplitudes.
#
# =============================================================

plt.figure(figsize=(10, 8))

for word in words:

    _, maxima = local_max_data[word]

    if len(maxima) < 2:
        continue

    plt.scatter(
        maxima[:-1],
        maxima[1:],
        s=25,
        alpha=0.6
    )


lims = plt.xlim()

plt.plot(
    lims,
    lims,
    linestyle="--"
)

plt.xlabel(
    "M(n)"
)

plt.ylabel(
    "M(n+1)"
)

plt.title(
    "Semantic Return Map"
)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.show()


# =============================================================
# 23. LYAPUNOV EXPONENT ESTIMATION
# =============================================================
#
# We estimate the largest Lyapunov exponent by following two
# nearby trajectories.
#
# If:
#
#       delta(t) ~ delta(0) exp(lambda t)
#
# then:
#
#       lambda > 0
#
# suggests local exponential divergence.
#
#       lambda < 0
#
# suggests convergence.
#
# NOTE:
#
# This is a numerical diagnostic for this constructed
# dynamical system. It is NOT a claim that the transformer
# embedding model itself has this Lyapunov exponent.
#
# =============================================================

def estimate_lyapunov(
    x0,
    category,
    epsilon=1e-6,
    steps=400
):

    direction = np.random.normal(
        size=len(x0)
    )

    direction = safe_normalize(
        direction
    )

    x1 = x0.copy()

    x2 = (
        x0
        +
        epsilon * direction
    )

    log_divergence = []

    for _ in range(steps):

        f1 = vector_field(
            x1,
            category
        )

        f2 = vector_field(
            x2,
            category
        )

        x1 = safe_normalize(
            x1 + DT * f1
        )

        x2 = safe_normalize(
            x2 + DT * f2
        )

        separation = np.linalg.norm(
            x2 - x1
        )

        if separation < 1e-15:
            continue

        log_divergence.append(
            np.log(
                separation / epsilon
            )
        )

        # Renormalize separation
        direction = (
            x2 - x1
        )

        direction = safe_normalize(
            direction
        )

        x2 = (
            x1
            +
            epsilon * direction
        )

    if len(log_divergence) == 0:
        return np.nan

    return (
        np.mean(log_divergence)
        / (steps * DT)
    )


print("\n")
print("=" * 70)
print("LYAPUNOV-EXPONENT ESTIMATES")
print("=" * 70)

lyapunov_values = {}

for category, group_words in categories.items():

    print(f"\n[{category.upper()}]")

    for word in group_words:

        x0 = embeddings[
            word_to_index[word]
        ]

        lam = estimate_lyapunov(
            x0,
            category
        )

        lyapunov_values[word] = lam

        print(
            f"{word:12s} "
            f"lambda ≈ {lam:.6f}"
        )


# =============================================================
# 24. LYAPUNOV BAR CHART
# =============================================================

plt.figure(figsize=(14, 7))

values = [
    lyapunov_values[w]
    for w in words
]

plt.bar(
    words,
    values
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    rotation=60
)

plt.ylabel(
    "Estimated largest Lyapunov exponent"
)

plt.title(
    "Local Stability of NLP Semantic Dynamics"
)

plt.tight_layout()

plt.show()


# =============================================================
# 25. BASIN OF ATTRACTION
# =============================================================
#
# We generate random starting points in the PCA plane.
#
# Each point is lifted back into the original embedding
# space approximately using the PCA inverse transform.
#
# Then we evolve the point under each candidate attractor
# and determine which attractor captures it.
#
# =============================================================

def classify_basin(
    x0,
    steps=300
):

    final_distances = {}

    for category in categories:

        trajectory = integrate(
            x0,
            category,
            steps=steps
        )

        final = trajectory[-1]

        distance = np.linalg.norm(
            final
            -
            attractors[category]
        )

        final_distances[category] = distance

    return min(
        final_distances,
        key=final_distances.get
    )


# =============================================================
# 26. BASIN GRID
# =============================================================
#
# Construct grid in PCA coordinates.
#
# =============================================================

mins = np.min(
    project(all_points),
    axis=0
)

maxs = np.max(
    project(all_points),
    axis=0
)

padding = 1.0

xmin = mins[0] - padding
xmax = maxs[0] + padding

ymin = mins[1] - padding
ymax = maxs[1] + padding


GRID = 35

gx = np.linspace(
    xmin,
    xmax,
    GRID
)

gy = np.linspace(
    ymin,
    ymax,
    GRID
)


basin_labels = []

print("\nComputing semantic basins...")

for y in gy:

    row = []

    for x in gx:

        point_2d = np.array([
            x,
            y
        ])

        # Approximate inverse PCA projection
        point_high_dim = (
            pca.inverse_transform(
                point_2d
            )
        )

        point_high_dim = safe_normalize(
            point_high_dim
        )

        basin = classify_basin(
            point_high_dim
        )

        row.append(
            basin
        )

    basin_labels.append(row)


# Convert categories to integers

category_names = list(
    categories.keys()
)

category_ids = {
    category: i
    for i, category
    in enumerate(category_names)
}


basin_numeric = np.array([
    [
        category_ids[c]
        for c in row
    ]
    for row in basin_labels
])


# =============================================================
# 27. VISUALIZE BASINS
# =============================================================

plt.figure(figsize=(12, 10))

plt.imshow(
    basin_numeric,
    origin="lower",
    extent=[
        xmin,
        xmax,
        ymin,
        ymax
    ],
    interpolation="nearest",
    alpha=0.25
)


for word in words:

    trajectory = projected_trajectories[word]

    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        linewidth=1.2,
        alpha=0.5
    )

    plt.text(
        trajectory[-1, 0],
        trajectory[-1, 1],
        word,
        fontsize=9
    )


for category, point in projected_attractors.items():

    plt.scatter(
        point[0],
        point[1],
        marker="*",
        s=400,
        edgecolors="black"
    )

    plt.text(
        point[0],
        point[1],
        f" {category}",
        fontweight="bold"
    )


for category, point in projected_repellors.items():

    plt.scatter(
        point[0],
        point[1],
        marker="X",
        s=220,
        edgecolors="black"
    )


plt.xlabel(
    "Semantic dimension 1"
)

plt.ylabel(
    "Semantic dimension 2"
)

plt.title(
    "Semantic Basins of Attraction"
)

plt.tight_layout()

plt.show()


# =============================================================
# 28. WORD-TO-WORD DISTANCE MATRIX
# =============================================================

from sklearn.metrics.pairwise import cosine_distances


distance_matrix = cosine_distances(
    embeddings
)


print("\n")
print("=" * 70)
print("WORD SEMANTIC DISTANCE MATRIX")
print("=" * 70)

print(
    " " * 12,
    " ".join(
        f"{w[:8]:>9s}"
        for w in words
    )
)

for i, word in enumerate(words):

    print(
        f"{word:12s}",
        " ".join(
            f"{distance_matrix[i,j]:9.3f}"
            for j in range(len(words))
        )
    )


# =============================================================
# 29. ATTRACTOR–WORD COSINE SIMILARITY
# =============================================================

print("\n")
print("=" * 70)
print("WORD → ATTRACTOR SIMILARITY")
print("=" * 70)

for word in words:

    x = embeddings[
        word_to_index[word]
    ]

    similarities = {}

    for category, attractor in attractors.items():

        similarities[category] = (
            np.dot(x, attractor)
            /
            (
                np.linalg.norm(x)
                *
                np.linalg.norm(attractor)
            )
        )

    ranking = sorted(
        similarities.items(),
        key=lambda z: z[1],
        reverse=True
    )

    print(
        f"\n{word}"
    )

    for category, score in ranking:

        print(
            f"    {category:10s}: "
            f"{score:.4f}"
        )


# =============================================================
# 30. ANIMATED TRAJECTORIES
# =============================================================
#
# This animation shows words moving through semantic space.
#
# =============================================================

fig, ax = plt.subplots(
    figsize=(12, 10)
)


# Determine global limits

all_projected = np.vstack(
    list(
        projected_trajectories.values()
    )
)

xlim = (
    all_projected[:, 0].min() - 1,
    all_projected[:, 0].max() + 1
)

ylim = (
    all_projected[:, 1].min() - 1,
    all_projected[:, 1].max() + 1
)


ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel(
    "Semantic dimension 1"
)

ax.set_ylabel(
    "Semantic dimension 2"
)

ax.set_title(
    "Animated NLP Attractor–Repellor Dynamics"
)


# -------------------------------------------------------------
# Create one point per word
# -------------------------------------------------------------

points = {}
labels = {}

for word in words:

    point, = ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6
    )

    label = ax.text(
        0,
        0,
        word,
        fontsize=9
    )

    points[word] = point
    labels[word] = label


# -------------------------------------------------------------
# Attractors
# -------------------------------------------------------------

for category, point in projected_attractors.items():

    ax.scatter(
        point[0],
        point[1],
        marker="*",
        s=350,
        edgecolors="black"
    )

    ax.text(
        point[0],
        point[1],
        category,
        fontweight="bold"
    )


# -------------------------------------------------------------
# Repellors
# -------------------------------------------------------------

for category, point in projected_repellors.items():

    ax.scatter(
        point[0],
        point[1],
        marker="X",
        s=200,
        edgecolors="black"
    )


def init():

    for word in words:

        points[word].set_data(
            [],
            []
        )

        labels[word].set_position(
            (0, 0)
        )

    return list(
        points.values()
    ) + list(
        labels.values()
    )


def update(frame):

    for word in words:

        trajectory = (
            projected_trajectories[word]
        )

        x = trajectory[frame, 0]
        y = trajectory[frame, 1]

        points[word].set_data(
            [x],
            [y]
        )

        labels[word].set_position(
            (x, y)
        )

    ax.set_title(
        "NLP Semantic Dynamics — "
        f"t = {frame * DT:.2f}"
    )

    return list(
        points.values()
    ) + list(
        labels.values()
    )


animation = FuncAnimation(
    fig,
    update,
    frames=range(
        0,
        STEPS + 1,
        5
    ),
    init_func=init,
    interval=30,
    blit=False
)

plt.show()


# =============================================================
# 31. SAVE ANIMATION
# =============================================================
#
# Uncomment one of these if desired.
#
# Requires ffmpeg for MP4.
#
# animation.save(
#     "nlp_attractor_repellor.mp4",
#     writer="ffmpeg",
#     fps=30
# )
#
# Or:
#
# animation.save(
#     "nlp_attractor_repellor.gif",
#     writer="pillow",
#     fps=20
# )


# =============================================================
# 32. SUMMARY
# =============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(
    """
The experiment constructed:

    Transformer word embeddings
             ↓
    Semantic attractors
             ↓
    Semantic repellors
             ↓
    Nonlinear dynamical system
             ↓
    Word trajectories
             ↓
    Phase portrait
             ↓
    Poincare sections
             ↓
    Local maxima
             ↓
    Return maps
             ↓
    Lyapunov diagnostics
             ↓
    Semantic basins
             ↓
    Animated dynamics

Remember:

The transformer provides the semantic state space.

The attractor–repellor differential equation defines
the dynamical system imposed on that state space.
"""
)
```

"""
---

# 2. The mathematical picture

The important part is the vector field:

$$
\dot{x}
=
\alpha(a-x)
+
\beta
\frac{x-r}
{\|x-r\|^2+\epsilon}
+
\eta\tanh(x)
-
\gamma x.
$$

Here \(x\) is a word embedding.

The first term

$$
\alpha(a-x)
$$

is the **semantic attraction**.

The second term

$$
\beta
\frac{x-r}
{\|x-r\|^2+\epsilon}
$$

is the **semantic repulsion**.

The nonlinear term

$$
\eta\tanh(x)
$$

makes the field nonlinear rather than simply being a linear spring system.

---

# 3. Semantic attractors

For example, the animal attractor is

$$
a_{\mathrm{animal}}
=
\frac{
x_{\mathrm{cat}}
+x_{\mathrm{dog}}
+x_{\mathrm{lion}}
+x_{\mathrm{tiger}}
+x_{\mathrm{wolf}}
+x_{\mathrm{horse}}
}
{6}.
$$

So the attractor is not itself a word.

It is a **semantic prototype**.

Conceptually:

```text
                     cat
                    /
             dog --+
                  /
              lion
                 \
                  +----> ANIMAL ATTRACTOR
                 /
             tiger
```

The same construction gives:

$$
a_{\mathrm{vehicle}},
\qquad
a_{\mathrm{food}},
\qquad
a_{\mathrm{emotion}}.
$$

---

# 4. Why the repellor is interesting

The repellor gives the system another geometric constraint.

Instead of simply saying

$$
x\rightarrow a,
$$

we have

$$
x\rightarrow a
\quad\text{while}\quad
x\not\rightarrow r.
$$

That produces a **flow field** rather than ordinary nearest-centroid classification.

This distinction becomes important when the attractor and repellor forces compete.

---

# 5. Basins of attraction

The basin experiment asks:

> If I start a word representation at this point, where does the dynamical system eventually take it?

So the semantic space becomes approximately:

```text
             ┌───────────────┐
             │ ANIMAL BASIN  │
             │       ↘       │
             │        ★      │
             └───────────────┘

          boundary / separatrix

             ┌───────────────┐
             │ VEHICLE BASIN │
             │       ↙       │
             │        ★      │
             └───────────────┘
```

The boundaries between these regions are potentially much more interesting than the clusters themselves.

---

# 6. Poincaré sections

For a continuous trajectory

$$
x(t),
$$

we select a hypersurface such as

$$
x_1=0.
$$

Every time the trajectory crosses that surface, we record its state.

Instead of examining

$$
x(t)
$$

directly, we get

$$
P_1,P_2,P_3,\ldots
$$

This converts continuous semantic dynamics into a discrete dynamical system.

---

# 7. Local maxima and return maps

Suppose a semantic coordinate produces

$$
M_1,M_2,M_3,\ldots
$$

at its local maxima.

The return map is

$$
M_{n+1}=F(M_n).
$$

This is particularly useful for detecting different dynamical regimes.

For example:

```text
Fixed point:

       •
      /
     /  y=x


Period-2:

       •
        \
         •


Chaotic-looking:

    •  •
      •   •
   •    •
      •    •
```

So the return map can reveal structure that is difficult to see from the original trajectory.

---

# 8. Lyapunov exponent

The code estimates

$$
\lambda
\approx
\frac{1}{T}
\log
\frac{\|\delta(T)\|}
{\|\delta(0)\|}.
$$

Interpretation:

$$
\lambda<0
$$

means nearby trajectories tend to converge.

$$
\lambda\approx0
$$

suggests marginal behavior.

$$
\lambda>0
$$

indicates local exponential separation and is one of the signatures one looks for in chaotic dynamics.

For this particular experiment, **do not interpret the measured value as a property of SBERT itself**. It is a property of the dynamical system we constructed using SBERT's embedding geometry.

---

# 9. The really interesting next step

The current program gives every word a prescribed target category:

```text
cat   → animal
dog   → animal
car   → vehicle
pizza → food
love  → emotion
```

A much richer experiment is to remove that assumption entirely.

Instead define

$$
\dot{x}_i
=
\sum_j
A_{ij}
\,f(x_j-x_i)
-
\sum_j
R_{ij}
\,g(x_j-x_i).
$$

Now **words interact with other words directly**.

For example:

$$
A_{ij}
=
\operatorname{sim}(x_i,x_j)
$$

and

$$
R_{ij}
=
1-\operatorname{sim}(x_i,x_j).
$$

Then the semantic system becomes something like:

```text
          dog
         ↗   ↘
      cat       lion
         ↘   ↗
          wolf

              ↓

        semantic attractor
```

while unrelated concepts exert repulsive forces.

That produces a genuine **many-body semantic dynamical system**:

$$
\boxed{
\dot X = F(X)
}
$$

where \(X\) contains all word states simultaneously.

That version is especially interesting because you can investigate whether **semantic attractors emerge spontaneously**, rather than specifying `"animal"` or `"vehicle"` beforehand.
"""