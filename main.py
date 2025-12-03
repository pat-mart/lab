import time

from src.graph import Graph
from src.tree_decomp import TreeDecomp


if __name__ == "__main__":
    usa_path = "include/road-road-usa/road-road-usa.mtx"
    mn_path = "include/road-minnesota/road-minnesota.mtx"
    us_48_path = "include/road-usroads-48/road-usroads-48.mtx"

    start_time = time.time()
    mn_graph = Graph(file_path=mn_path)
    # usa_graph = Graph(file_path=usa_path)
    # usa_48_graph = Graph(file_path=us_48_path)

    end_time = time.time()

    parse_time = end_time - start_time

    print("Graph parse time: " + str(parse_time))

    start_time = time.time()
    bfs_result = Graph.bfs(mn_graph.g, 7)
    end_time = time.time()

    bfs_time = end_time - start_time

    print("BFS time: " + str(bfs_time))
    print("BFS size: " + str(len(bfs_result[0])))
    print("Graph diameter (lower, upper): " + str(mn_graph.estimate_diameter(samples=10)))
    print("Total time: " + str(parse_time + bfs_time))

    start_time = time.time()
    td_bags, td_adj, root = mn_graph.dp_tree_decomp()
    end_time = time.time()
    
    tw = 0
    for root, val in td_bags.items():
        tw = max(tw, len(val))

    print("DP tree decomposition time: " + str(end_time - start_time))
    print("Treewidth: " + str(tw - 1))
    
    pos, dis = mn_graph.h_two_h(td_bags, td_adj, root)
    print(dis)