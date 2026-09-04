import math
import random
from itertools import permutations

# generated from ChatGPT using "Generate me a good tutorial example for Weiqi Li. 2024. Optimizing with Attractor: A Tutorial. ACM Comput. Surv. 56, 9, Article 235 (September 2024), 41 pages. https://doi.org/10.1145/3648354"

# ============================================================
# 1. TSP instance
# ============================================================

cities = {
    0: (0.0, 0.0),
    1: (1.0, 3.0),
    2: (4.0, 1.0),
    3: (6.0, 4.0),
    4: (7.0, 0.0),
    5: (9.0, 3.0),
    6: (8.0, 7.0),
    7: (5.0, 8.0),
    8: (2.0, 7.0),
    9: (0.0, 5.0),
}


def distance(a, b):
    x1, y1 = cities[a]
    x2, y2 = cities[b]
    return math.hypot(x1 - x2, y1 - y2)


N = len(cities)

D = [[distance(i, j) for j in range(N)] for i in range(N)]


# ============================================================
# 2. Tour cost
# ============================================================

def tour_cost(tour):
    """Cost of a Hamiltonian cycle."""
    return sum(
        D[tour[i]][tour[(i + 1) % N]]
        for i in range(N)
    )


# ============================================================
# 3. 2-opt local search
#
#    This is the dynamical-system component:
#
#        tour -> better neighboring tour -> ...
#
#    until we reach a local optimum.
# ============================================================

def two_opt(tour):
    tour = list(tour)

    improved = True

    while improved:
        improved = False
        current_cost = tour_cost(tour)

        for i in range(1, N - 2):
            for j in range(i + 1, N):

                candidate = (
                    tour[:i]
                    + tour[i:j][::-1]
                    + tour[j:]
                )

                candidate_cost = tour_cost(candidate)

                if candidate_cost < current_cost - 1e-12:
                    tour = candidate
                    current_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

    return tuple(tour)


# ============================================================
# 4. Multi-start local search
#
#    Each random initial tour is a search trajectory.
#    The endpoint is a locally optimal solution.
# ============================================================

def multi_start_local_search(K=100, seed=0):
    random.seed(seed)

    local_optima = []

    for _ in range(K):

        tour = list(range(N))
        random.shuffle(tour)

        optimum = two_opt(tour)

        local_optima.append(optimum)

    return local_optima


# ============================================================
# 5. Construct the ATTRACTOR
#
#    E[i][j] = number of locally optimal tours containing
#              edge (i,j).
#
#    This is the important idea:
#
#        solution space
#             |
#             | local search
#             v
#        locally optimal tours
#             |
#             | collect common edges
#             v
#          ATTRACTOR
# ============================================================

def build_attractor(local_optima):
    E = [[0 for _ in range(N)] for _ in range(N)]

    for tour in local_optima:

        for i in range(N):
            a = tour[i]
            b = tour[(i + 1) % N]

            # Undirected TSP edge
            E[a][b] += 1
            E[b][a] += 1

    return E


# ============================================================
# 6. Inspect attractor
# ============================================================

def print_attractor(E, threshold=1):

    print("\nAttractor edges")
    print("----------------")

    for i in range(N):
        for j in range(i + 1, N):

            if E[i][j] >= threshold:
                print(
                    f"{i} -- {j} : "
                    f"{E[i][j]} / {len(local_optima)}"
                )


# ============================================================
# 7. Exhaustive search INSIDE the attractor
#
#    Instead of considering every possible edge,
#    only edges contained in the attractor are allowed.
# ============================================================

def exhaustive_attractor_search(E):

    start = 0

    best_tour = None
    best_cost = float("inf")

    # Fix city 0 as the first city to remove rotational duplicates.
    remaining = set(range(1, N))

    nodes_examined = 0

    def dfs(path, unused):

        nonlocal best_tour
        nonlocal best_cost
        nonlocal nodes_examined

        if not unused:

            # Need to return to starting city.
            if E[path[-1]][start] == 0:
                return

            nodes_examined += 1

            cost = tour_cost(path)

            if cost < best_cost:
                best_cost = cost
                best_tour = tuple(path)

            return

        current = path[-1]

        for nxt in unused:

            # Attractor restriction
            if E[current][nxt] == 0:
                continue

            dfs(
                path + [nxt],
                unused - {nxt}
            )

    dfs([start], remaining)

    return best_tour, best_cost, nodes_examined


# ============================================================
# 8. Ordinary exhaustive search
#
#    Used only for verification on this small example.
# ============================================================

def ordinary_exhaustive_search():

    best_tour = None
    best_cost = float("inf")

    count = 0

    for perm in permutations(range(1, N)):

        tour = (0,) + perm

        count += 1

        cost = tour_cost(tour)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost, count


# ============================================================
# 9. Run the attractor-based algorithm
# ============================================================

if __name__ == "__main__":

    print("=== ATTRACTOR-BASED SEARCH ===")

    # Phase 1: local search
    local_optima = multi_start_local_search(
        K=200,
        seed=42
    )

    print(
        f"Generated {len(local_optima)} "
        f"local-search trajectories."
    )

    print(
        f"Distinct local optima: "
        f"{len(set(local_optima))}"
    )

    # Phase 2: construct attractor
    E = build_attractor(local_optima)

    print_attractor(E, threshold=10)

    # Phase 3: exhaustive search inside attractor
    attractor_tour, attractor_cost, attractor_count = \
        exhaustive_attractor_search(E)

    print("\nAttractor search")
    print("----------------")
    print("Tour :", attractor_tour)
    print("Cost :", attractor_cost)
    print(
        "Complete tours examined inside attractor:",
        attractor_count
    )

    # ========================================================
    # Verification
    # ========================================================

    exact_tour, exact_cost, exact_count = \
        ordinary_exhaustive_search()

    print("\nFull exhaustive search")
    print("----------------------")
    print("Tour :", exact_tour)
    print("Cost :", exact_cost)
    print(
        "Complete tours examined:",
        exact_count
    )

    print("\nVerification")
    print("------------")

    if abs(attractor_cost - exact_cost) < 1e-12:
        print("SUCCESS: attractor search found the exact optimum.")
    else:
        print(
            "The attractor did not contain the optimum "
            "with this sampling configuration."
        )