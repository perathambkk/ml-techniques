"""
===============================================================
Attractor-Based Search Tutorial
Inspired by:

WeiQi Li. 2024.
"Optimizing with Attractor: A Tutorial."
ACM Computing Surveys, 56(9), Article 235.

Tutorial example:
    Symmetric Euclidean Traveling Salesman Problem (TSP)

The program demonstrates:

    1. Multi-start local search
    2. Convergence to local optima
    3. Construction of edge-frequency matrix E
    4. Construction of an attractor
    5. Attractor-restricted exhaustive search
    6. Full exhaustive search for comparison
    7. Printing of the entire restricted search tree
    8. Visualization of local-search trajectories
===============================================================
"""

import math
import random
from itertools import permutations

import matplotlib.pyplot as plt

# generated from ChatGPT using "Generate me a good tutorial example for Weiqi Li. 2024. Optimizing with Attractor: A Tutorial. ACM Comput. Surv. 56, 9, Article 235 (September 2024), 41 pages. https://doi.org/10.1145/3648354" -> "make a version that prints the entire search tree, the edge matrix E****, the number of tours before/after attractor reduction, and visually shows the local-search trajectories converging into the attractor. go ahead"


# =============================================================
# 1. TSP INSTANCE
# =============================================================

CITIES = {
    0: (0.0, 0.0),
    1: (2.0, 5.0),
    2: (5.0, 7.0),
    3: (8.0, 6.0),
    4: (10.0, 2.0),
    5: (7.0, 0.0),
    6: (3.0, 1.0),
    7: (1.0, 3.0),
}

N = len(CITIES)


def euclidean_distance(a, b):
    x1, y1 = CITIES[a]
    x2, y2 = CITIES[b]
    return math.hypot(x1 - x2, y1 - y2)


DIST = [
    [euclidean_distance(i, j) for j in range(N)]
    for i in range(N)
]


# =============================================================
# 2. TOUR UTILITIES
# =============================================================

def tour_cost(tour):
    """
    Cost of a Hamiltonian cycle.

    Example:

        0 -> 1 -> 2 -> ... -> 0
    """
    return sum(
        DIST[tour[i]][tour[(i + 1) % N]]
        for i in range(N)
    )


def canonical_tour(tour):
    """
    Because a TSP cycle can be rotated, normalize it
    so city 0 is always first.

    We also choose the lexicographically smaller orientation.
    """

    tour = list(tour)

    # Rotate so that 0 comes first
    k = tour.index(0)

    tour = tour[k:] + tour[:k]

    reverse = [tour[0]] + list(reversed(tour[1:]))

    return tuple(min(tour, reverse))


def tour_edges(tour):
    """
    Return the undirected edges of a tour.
    """

    edges = set()

    for i in range(N):
        a = tour[i]
        b = tour[(i + 1) % N]

        edge = tuple(sorted((a, b)))

        edges.add(edge)

    return edges


# =============================================================
# 3. 2-OPT NEIGHBORHOOD
# =============================================================

def two_opt_neighbors(tour):
    """
    Generate all 2-opt neighbors.

    A 2-opt move removes two edges and reconnects
    the resulting paths in the opposite way.
    """

    tour = list(tour)

    for i in range(1, N - 2):

        for j in range(i + 1, N):

            candidate = (
                tour[:i]
                + tour[i:j][::-1]
                + tour[j:]
            )

            yield tuple(candidate)


# =============================================================
# 4. LOCAL SEARCH
# =============================================================

def local_search(initial_tour, trajectory_id=None):
    """
    First-improvement 2-opt local search.

    Returns:

        local optimum
        trajectory

    The trajectory contains every solution visited.
    """

    current = canonical_tour(initial_tour)

    trajectory = [
        (current, tour_cost(current))
    ]

    while True:

        current_cost = tour_cost(current)

        found_better = False

        for candidate in two_opt_neighbors(current):

            candidate = canonical_tour(candidate)

            candidate_cost = tour_cost(candidate)

            if candidate_cost < current_cost - 1e-12:

                current = candidate

                trajectory.append(
                    (current, candidate_cost)
                )

                found_better = True

                break

        if not found_better:
            break

    return current, trajectory


# =============================================================
# 5. MULTI-START SEARCH
# =============================================================

def multi_start_search(number_of_starts=30, seed=42):

    random.seed(seed)

    all_trajectories = []
    local_optima = []

    for run in range(number_of_starts):

        initial = list(range(N))

        random.shuffle(initial)

        optimum, trajectory = local_search(
            initial,
            trajectory_id=run
        )

        local_optima.append(optimum)

        all_trajectories.append(trajectory)

    return local_optima, all_trajectories


# =============================================================
# 6. BUILD EDGE-FREQUENCY MATRIX E
# =============================================================

def build_edge_matrix(local_optima):

    E = [
        [0 for _ in range(N)]
        for _ in range(N)
    ]

    for tour in local_optima:

        for a, b in tour_edges(tour):

            E[a][b] += 1
            E[b][a] += 1

    return E


# =============================================================
# 7. PRINT MATRIX E
# =============================================================

def print_edge_matrix(E, number_of_samples):

    print()
    print("=" * 72)
    print("EDGE-FREQUENCY MATRIX E")
    print("=" * 72)

    print()
    print("E[i][j] = number of locally optimal tours containing edge i-j")
    print()

    print("       " + " ".join(f"{j:5d}" for j in range(N)))

    for i in range(N):

        print(
            f"{i:3d} : "
            + " ".join(
                f"{E[i][j]:5d}"
                for j in range(N)
            )
        )

    print()
    print(
        f"Number of sampled local optima = {number_of_samples}"
    )


# =============================================================
# 8. CONSTRUCT ATTRACTOR
# =============================================================

def attractor_edges(E, threshold):

    edges = set()

    for i in range(N):

        for j in range(i + 1, N):

            if E[i][j] >= threshold:

                edges.add((i, j))

    return edges


def print_attractor(E, threshold, number_of_samples):

    edges = attractor_edges(E, threshold)

    print()
    print("=" * 72)
    print("ATTRACTOR")
    print("=" * 72)

    print()
    print(
        f"Threshold = {threshold}/{number_of_samples}"
    )

    print()
    print("Edge        Frequency       Relative frequency")
    print("-" * 52)

    for a, b in sorted(edges):

        frequency = E[a][b]

        print(
            f"{a} -- {b}"
            f"{frequency:14d}"
            f"{frequency / number_of_samples:20.2%}"
        )

    print()
    print(
        f"Number of attractor edges = {len(edges)}"
    )

    return edges


# =============================================================
# 9. TEST WHETHER A TOUR IS INSIDE THE ATTRACTOR
# =============================================================

def tour_is_inside_attractor(tour, attractor):

    return tour_edges(tour).issubset(attractor)


# =============================================================
# 10. PRINT SEARCH TREE
# =============================================================

def print_tree_node(path, unused, attractor, depth=0):

    indent = "    " * depth

    current = path[-1]

    if not unused:

        closing_edge = tuple(
            sorted((current, 0))
        )

        if closing_edge in attractor:

            complete = tuple(path)

            print(
                indent
                + f"└── COMPLETE {complete}"
                + f"   cost={tour_cost(complete):.3f}"
            )

        else:

            print(
                indent
                + f"└── PRUNED return edge "
                f"{current}-0 not in attractor"
            )

        return

    for nxt in sorted(unused):

        edge = tuple(
            sorted((current, nxt))
        )

        if edge not in attractor:

            print(
                indent
                + f"├── {nxt}"
                + f"  [PRUNE: edge {current}-{nxt}]"
            )

            continue

        print(
            indent
            + f"├── {nxt}"
            + f"  [edge {current}-{nxt} allowed]"
        )

        print_tree_node(
            path + [nxt],
            unused - {nxt},
            attractor,
            depth + 1
        )


def print_search_tree(attractor):

    print()
    print("=" * 72)
    print("ATTRACTOR-RESTRICTED SEARCH TREE")
    print("=" * 72)

    print()
    print("Root = city 0")
    print()
    print_tree_node(
        [0],
        set(range(1, N)),
        attractor
    )


# =============================================================
# 11. EXHAUSTIVE SEARCH INSIDE ATTRACTOR
# =============================================================

def attractor_exhaustive_search(attractor):

    best_tour = None
    best_cost = float("inf")

    complete_tours = []
    nodes = 0
    pruned = 0

    def dfs(path, unused):

        nonlocal best_tour
        nonlocal best_cost
        nonlocal nodes
        nonlocal pruned

        current = path[-1]

        # -----------------------------------------------------
        # Complete path
        # -----------------------------------------------------

        if not unused:

            closing_edge = tuple(
                sorted((current, 0))
            )

            if closing_edge not in attractor:

                pruned += 1

                return

            nodes += 1

            tour = tuple(path)

            cost = tour_cost(tour)

            complete_tours.append(
                (tour, cost)
            )

            if cost < best_cost:

                best_cost = cost
                best_tour = tour

            return

        # -----------------------------------------------------
        # Expand node
        # -----------------------------------------------------

        for nxt in sorted(unused):

            edge = tuple(
                sorted((current, nxt))
            )

            if edge not in attractor:

                pruned += 1

                continue

            dfs(
                path + [nxt],
                unused - {nxt}
            )

    dfs([0], set(range(1, N)))

    return (
        best_tour,
        best_cost,
        nodes,
        pruned,
        complete_tours
    )


# =============================================================
# 12. FULL EXHAUSTIVE SEARCH
# =============================================================

def full_exhaustive_search():

    best_tour = None
    best_cost = float("inf")

    count = 0

    for permutation in permutations(range(1, N)):

        tour = (0,) + permutation

        count += 1

        cost = tour_cost(tour)

        if cost < best_cost:

            best_cost = cost
            best_tour = tour

    return best_tour, best_cost, count


# =============================================================
# 13. VISUALIZATION
# =============================================================

def plot_cities():

    plt.figure(figsize=(8, 7))

    for city, (x, y) in CITIES.items():

        plt.scatter(x, y, s=100)

        plt.text(
            x + 0.15,
            y + 0.15,
            str(city),
            fontsize=12
        )

    plt.title("TSP Instance")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.axis("equal")
    plt.grid(True)

    plt.show()


def plot_local_search_trajectories(
    trajectories,
    local_optima,
    attractor
):
    """
    Visualize each city-to-city edge as a point in
    an abstract 'solution-space' representation.

    x = tour cost
    y = trajectory/run number

    Every local-search trajectory therefore moves
    toward a local optimum.

    Local optima belonging to the attractor are emphasized.
    """

    plt.figure(figsize=(11, 7))

    for run, trajectory in enumerate(trajectories):

        costs = [
            cost
            for _, cost in trajectory
        ]

        xs = list(range(len(costs)))

        plt.plot(
            xs,
            costs,
            marker="o",
            alpha=0.45
        )

    # Mark final local optima
    for run, optimum in enumerate(local_optima):

        cost = tour_cost(optimum)

        plt.scatter(
            len(trajectories[run]) - 1,
            cost,
            s=100,
            marker="*"
        )

    plt.title(
        "Local-search trajectories converging to local optima"
    )

    plt.xlabel("Local-search iteration")
    plt.ylabel("Tour cost")

    plt.grid(True)

    plt.show()


def plot_attractor_tour(
    best_tour,
    attractor
):

    plt.figure(figsize=(8, 7))

    # Draw attractor edges
    for a, b in attractor:

        xa, ya = CITIES[a]
        xb, yb = CITIES[b]

        plt.plot(
            [xa, xb],
            [ya, yb],
            linewidth=1.5,
            alpha=0.35
        )

    # Draw best tour
    if best_tour is not None:

        closed = list(best_tour) + [
            best_tour[0]
        ]

        xs = [
            CITIES[c][0]
            for c in closed
        ]

        ys = [
            CITIES[c][1]
            for c in closed
        ]

        plt.plot(
            xs,
            ys,
            linewidth=3,
            marker="o"
        )

    # Cities
    for city, (x, y) in CITIES.items():

        plt.scatter(
            x,
            y,
            s=120
        )

        plt.text(
            x + 0.15,
            y + 0.15,
            str(city),
            fontsize=12
        )

    plt.title(
        "Attractor edges and best tour"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.axis("equal")
    plt.grid(True)

    plt.show()


# =============================================================
# 14. MAIN EXPERIMENT
# =============================================================

def main():

    print()
    print("#" * 72)
    print("ATTRACTOR-BASED SEARCH TUTORIAL")
    print("#" * 72)

    print()
    print("Number of cities:", N)

    # ---------------------------------------------------------
    # Visualize original problem
    # ---------------------------------------------------------

    plot_cities()

    # ---------------------------------------------------------
    # Phase 1:
    # Multi-start local search
    # ---------------------------------------------------------

    NUMBER_OF_STARTS = 30

    local_optima, trajectories = \
        multi_start_search(
            number_of_starts=NUMBER_OF_STARTS,
            seed=42
        )

    print()
    print("=" * 72)
    print("PHASE 1 — MULTI-START LOCAL SEARCH")
    print("=" * 72)

    for run, trajectory in enumerate(trajectories):

        initial_tour = trajectory[0][0]
        final_tour = trajectory[-1][0]

        print()
        print(f"Run {run + 1:02d}")

        print(
            f"  initial: {initial_tour}"
            f"   cost={tour_cost(initial_tour):.3f}"
        )

        print("  trajectory:")

        for step, (tour, cost) in enumerate(trajectory):

            print(
                f"      {step:02d}: "
                f"{tour}"
                f"   cost={cost:.3f}"
            )

        print(
            f"  local optimum: {final_tour}"
            f"   cost={tour_cost(final_tour):.3f}"
        )

    # ---------------------------------------------------------
    # Show convergence
    # ---------------------------------------------------------

    plot_local_search_trajectories(
        trajectories,
        local_optima,
        None
    )

    # ---------------------------------------------------------
    # Distinct local optima
    # ---------------------------------------------------------

    unique_optima = sorted(
        set(local_optima),
        key=tour_cost
    )

    print()
    print("=" * 72)
    print("LOCAL OPTIMA")
    print("=" * 72)

    for i, tour in enumerate(unique_optima):

        frequency = local_optima.count(tour)

        print(
            f"{i + 1:02d}. "
            f"{tour}"
            f"   cost={tour_cost(tour):.3f}"
            f"   frequency={frequency}"
        )

    # ---------------------------------------------------------
    # Phase 2:
    # Construct E
    # ---------------------------------------------------------

    E = build_edge_matrix(local_optima)

    print_edge_matrix(
        E,
        len(local_optima)
    )

    # ---------------------------------------------------------
    # Construct attractor
    # ---------------------------------------------------------

    # For this tutorial we require an edge to occur in
    # at least 20% of the locally optimal tours.

    threshold = max(
        1,
        math.ceil(0.20 * NUMBER_OF_STARTS)
    )

    attractor = print_attractor(
        E,
        threshold,
        NUMBER_OF_STARTS
    )

    # ---------------------------------------------------------
    # Phase 3:
    # Restricted exhaustive search
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 3 — EXHAUSTIVE SEARCH INSIDE ATTRACTOR")
    print("=" * 72)

    (
        attractor_tour,
        attractor_cost,
        complete_count,
        pruned_count,
        complete_tours
    ) = attractor_exhaustive_search(
        attractor
    )

    print()
    print("Best attractor tour:")
    print(" ", attractor_tour)

    print(
        f"Cost = {attractor_cost:.6f}"
    )

    print()
    print(
        "Complete tours examined inside attractor:",
        complete_count
    )

    print(
        "Branches pruned by attractor:",
        pruned_count
    )

    # ---------------------------------------------------------
    # Print the search tree
    # ---------------------------------------------------------

    print_search_tree(attractor)

    # ---------------------------------------------------------
    # Phase 4:
    # Full exhaustive search
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 4 — FULL EXHAUSTIVE SEARCH")
    print("=" * 72)

    exact_tour, exact_cost, full_count = \
        full_exhaustive_search()

    print()
    print("Exact optimal tour:")
    print(" ", exact_tour)

    print(
        f"Cost = {exact_cost:.6f}"
    )

    print()
    print(
        "Complete tours examined:",
        full_count
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("SEARCH-SPACE COMPARISON")
    print("=" * 72)

    reduction = (
        1.0
        - complete_count / full_count
    )

    print()
    print(
        f"Full search space       : {full_count:,}"
    )

    print(
        f"Attractor search space  : {complete_count:,}"
    )

    print(
        f"Reduction               : {reduction:.2%}"
    )

    if full_count > 0:

        print(
            f"Search-space ratio      : "
            f"{complete_count / full_count:.4f}"
        )

    # ---------------------------------------------------------
    # Correctness check
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("CORRECTNESS CHECK")
    print("=" * 72)

    if (
        attractor_tour is not None
        and abs(attractor_cost - exact_cost) < 1e-9
    ):

        print()
        print(
            "✓ Attractor-restricted search found "
            "the global optimum."
        )

    else:

        print()
        print(
            "⚠ The sampled attractor did not contain "
            "the global optimum."
        )

        print(
            "Increase NUMBER_OF_STARTS or lower "
            "the attractor threshold."
        )

    # ---------------------------------------------------------
    # Final visualization
    # ---------------------------------------------------------

    plot_attractor_tour(
        exact_tour,
        attractor
    )


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":
    main()