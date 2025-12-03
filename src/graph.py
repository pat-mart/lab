import math
import copy
from collections import deque
import random
from src.tree_decomp import TreeDecomp

class Graph:
    
    def __init__(self, adjacency_dict: dict[int, list[int]]=None, file_path=None):

        if adjacency_dict:
            self.g: dict[int, list[int]] = adjacency_dict
        elif file_path:
            self.g, self.start_key = self.parse_mtx(file_path)
        else:
            self.g = {}
            self.start_key = None
            
        self.edges = self.get_edges(adj=self.g)
    
    @staticmethod
    def get_edges(adj: dict[int, list[int]]) -> set[tuple]:
        edges = set()
        
        for k, v in adj.items():
            for vertex in v:
                edges.add((k, vertex))
        
        return edges

    @staticmethod
    def min_degree(h: dict[int, list[int]]) -> int:
        min_k = list(h.keys())[0]
        for k, v in h.items():
            if len(v) < len(h[min_k]):
                min_k = k
        
        return min_k
    
    @staticmethod
    def max_degree(h: dict[int, list[int]]) -> int:
        max_k = list(h.keys())[0]
        for k, v in h.items():
            if len(v) > len(h[max_k]):
                max_k = k
        
        return max_k

    @staticmethod
    def parse_mtx(file_path: str):
        adjacency_dict = {}
        start_key = None

        with open(file_path, 'r') as file:
            for line in file:
                if not line.strip() or line.startswith('%'):
                    continue

                parts = line.strip().split()
                if len(parts) != 2:
                    continue

                u, v = map(int, parts)
                if start_key is None:
                    start_key = u

                adjacency_dict.setdefault(u, []).append(v)
                adjacency_dict.setdefault(v, []).append(u)

        return adjacency_dict, start_key

    @staticmethod
    def bfs(g: dict[int, list[int]], start_key: int, goal_key: int = -1):
        visited = set()
        queue = deque([start_key])
        order = []
        distances = {start_key: 0}

        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                order.append(vertex)

                current_distance = distances[vertex]

                if goal_key != -1 and goal_key == vertex:
                    return order, distances

                for neighbor in g[vertex]:
                    if neighbor not in visited:
                        distances[neighbor] = current_distance + 1
                        queue.append(neighbor)

        return order, distances
    
    '''Returns adjacency dict consisting of vertex and its neighbors as vertices, edges from vertex to its neighbors as edges'''
    @staticmethod
    def star(adj: dict[int, list[int]], vertex: int) -> dict[int, list[int]]:
        if not vertex in adj:
            return {}
        
        return {vertex: adj[vertex]}

    def neighbors(self, vertex: int) -> list[int]:
        return self.star(self.g, vertex)

    def diameter(self, sample_size=100):
        d = 0
        vertices = list(self.g.keys())

        for i in range(min(len(vertices), sample_size)):
            start_vertex = vertices[i]
            dists = self.bfs(start_vertex)[1]
            d = max(d, max(dists.values()))
        return d
    
    def estimate_diameter(self, samples=10):
        vertices = list(self.g.keys())
        
        upper_bound = math.inf
        lower_bound = 0

        for _ in range(samples):
            start = random.choice(vertices)
            ecc = max(self.bfs(self.g, start)[1].values())
            
            lower_bound = max(lower_bound, ecc)
            upper_bound = min(upper_bound, ecc * 2)

        return lower_bound, upper_bound
    
    # Algorithm 2 - eliminates vertices in g using min degree
    def vertex_elim(self, adj: dict[int, list[int]], vertex: int) -> dict[int, list[int]]:
        if vertex not in adj:
            return {u: list(neis) for u, neis in adj.items()}
        
        working: dict[int, set[int]] = {u: set(neis) for u, neis in adj.items()}

        new_h: dict[int, set[int]] = {u: set(neis) for u, neis in working.items()}
        
        neighbors = list(working[vertex])
        
        for i in range(len(neighbors)):
            u = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if w not in new_h[u]:
                    new_h[u].add(w)
                    new_h[w].add(u)

        for u in neighbors:
            new_h[u].discard(vertex)
            
        del new_h[vertex]

        return {u: sorted(list(neis)) for u, neis in new_h.items()}

    
    '''Returns tree decomposition bags with [bag root: vertices], tree decomposition adjacency with [bag root: neighbor bag roots], root of tree decomposition (root of last bag added)'''
    def dp_tree_decomp(self):
        h_adj = copy.deepcopy(self.g)
        
        td_bags: dict[int, list[int]] = {} # key: bag root, value: all vertices in bag
        td_adj: dict[int, list[int]] = {} # key: bag root, value: adjacent bag roots in td
        
        ordering: dict[int, int] = {}
        num_vertices = len(self.g)
        
        root = 0

        for i in range(1, num_vertices+1):
            min_degree: int = self.min_degree(h=h_adj)
            star_min_deg = self.star(h_adj, min_degree)
            
            td_bags[min_degree] = self.dict_to_list(star_min_deg)
            
            h_adj = self.vertex_elim(h_adj, min_degree)
            ordering[min_degree] = i
            
        td_adj = {v: [] for v in td_bags}

        for v in td_bags:
            # X(v)
            bag = td_bags[v]

            # find vertex in X(v)\{v} with smallest elimination order
            candidates = [u for u in bag if u != v]
            if not candidates:
                continue

            min_u = min(candidates, key=lambda u: ordering[u])
            
            td_adj[v].append(min_u)
            td_adj[min_u].append(v)
                        
        # addl for loop here for reassigning edge weights
        
        return td_bags, td_adj, root

    @staticmethod
    def dict_to_list(adj: dict[int, list[int]]) -> list[int]:
        to_ret = [list(adj.keys())[0]]
        for value in adj.values():
            for v in value:
                to_ret.append(v)

        return to_ret

    # Hierarchical 2-hop indexing
    def h_two_h(self, td_bags, td_adj, root):
        
        # implement algorithm 1
        # find graph of ~100k vertices
        # make tables similar to figure 11
        
        top_down = self.bfs(td_adj, root)[0]
        
        pos: dict[int, list[int]] = {}
        dis: dict[int, list[int]] = {}
        anc: dict[int, list[int]] = {}
        
        # precompute anc list for every vertex
        for bag_root in top_down:
            anc_list = self.bfs(td_adj, root, bag_root)[0]
            
            if not bag_root in anc:
                anc[bag_root] = anc_list
        
        # precompute distance arrays 
        # array of distances (from original graph) of root of X(v) to every vertex in X(v).anc

        for bag_root in top_down:
            
            vertex_list = td_bags[bag_root]
            
            pos_v = []
            for v in vertex_list:
                pos_v.append(anc[bag_root].index(v) + 1)
                
            pos[bag_root] = pos_v
            anc_list = anc[bag_root]
            
            dis_v = [float('inf')] * len(anc_list)
            
            
            for i in range(len(anc_list) - 1):
                best_distance = float('inf')
                
                for j in range(len(td_bags[bag_root]) - 1):
                    
                    if vertex_list[j] == bag_root:
                        continue
                    
                    if pos[bag_root][j] > i:
                        xj = td_bags[bag_root][j]
                        if xj in dis: 
                            temp_dist = dis[xj][i]
                        else:
                            temp_dist = float('inf')
                    else: 
                        anc_i = anc_list[i]
                        pos_xj = pos[bag_root][j]
                        
                        if anc_i in dis: 
                            temp_dist = dis[anc_i][pos_xj]
                        else: 
                            temp_dist = float('inf')
                
                    total = 1 + temp_dist
                    best_distance = min(total, best_distance)
                    
                dis_v[i] = best_distance
            
            dis[bag_root] = dis_v
            dis_v[len(anc_list) - 1] = 0

        return pos, dis