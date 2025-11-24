from collections import deque
import random
import time

from src.graph import Graph
from src.tree_decomp import TreeDecomp


if __name__ == "__main__":
    usa_path = "include/road-road-usa.mtx"
    mn_path = "include/road-minnesota.mtx"
    lux_path = "include/road-luxembourg-osm.mtx"

    print("\nLoading graph")
    start_time = time.time()
    G = Graph(file_path=mn_path)
    parse_time = time.time() - start_time
    print(f"Graph parse time: {parse_time:.4f} seconds")

    print("\nRunning BFS sample")
    start_time = time.time()
    bfs_order, bfs_dist = G.bfs(G.start_key)
    bfs_time = time.time() - start_time
    print(f"BFS time: {bfs_time:.4f} seconds")
    print(f"BFS reachable nodes: {len(bfs_order)}")
    print("Estimated diameter (lower, upper):", G.estimate_diameter(samples=10))

    print("\nAlgorithm 3")
    start_time = time.time()
    bags, lambdas, parent, phi = G.naive_dp_tree_decomp(G.g)
    td_time = time.time() - start_time
    print(f"DP Tree Decomposition time: {td_time:.4f} seconds")
    print(f"Number of bags: {len(bags)}")

    print("\nAlgorithm 5")
    start_time = time.time()
    anc, pos, dis = G.naive_H2H(bags, lambdas, parent, phi)
    h2h_time = time.time() - start_time
    print(f"H2H index time: {h2h_time:.4f} seconds")

    print("\nStart vertex tables")
    s = G.start_key
    if s in bags:
        print("bag[start]:", bags[s])
        print("lambda[start]:", lambdas[s])
        print("anc[start]:", anc[s])
        print("pos[start]:", pos[s])
        print("dis[start]:", dis[s])
    else:
        print("Start vertex not found in bags.")

    print("\nAlgorithm 1")

    verts = list(G.g.keys())
    random.seed(0)

    # Run 5 random distance queries
    for _ in range(5):
        u = random.choice(verts)
        v = random.choice(verts)

        # H2H distance
        h2h_dist = G.H2H_query(u, v, anc, pos, dis, bags)

        # BFS ground truth
        _, bfs_from_u = G.bfs(u)
        bfs_true = bfs_from_u.get(v, float("inf"))

        print(f"\nQuery: dist({u}, {v})")
        print(f"  H2H distance: {h2h_dist}")
        print(f"  BFS distance: {bfs_true}")

