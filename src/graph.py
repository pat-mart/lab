import math
import copy
from collections import deque
from itertools import combinations
import random

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
    def bfs(g: dict[int, list[int]], start_key: int):
        visited = set([start_key])
        queue = deque([start_key])
        order = []
        distances = {start_key: 0}

        while queue:
            vertex = queue.popleft()
            order.append(vertex)

            current_distance = distances[vertex]

            for neighbor in g[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)  
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)

        return order, distances
    
    @staticmethod
    def bfs_shortest_path(g: dict[int, list[int]], start_key: int, goal_key: int):
        visited = set([start_key])
        queue = deque([start_key])
        
        distances = {start_key: 0}
        parent = {start_key: None}

        while queue:
            v = queue.popleft()

            if v == goal_key:
                path = []
                cur = v
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path, distances
            
            for u in g[v]:
                if u not in visited:
                    visited.add(u)
                    parent[u] = v
                    distances[u] = distances[v] + 1
                    queue.append(u)
                    
        return None, distances
    
    @staticmethod
    def dfs(g: dict[int, list[int]], start_key: int):
        order = []
        stack = [start_key]
        parent = {start_key: None}

        while stack:
            v = stack.pop()
            order.append(v)
            for u in g[v]:
                if u not in parent:
                    parent[u] = v
                    stack.append(u)

        return order
    
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
    
    @staticmethod
    def vertex_elim(adj: dict[int, list[int]], vertex: int) -> dict[int, list[int]]:
        if vertex not in adj:
            return {u: list(neis) for u, neis in adj.items()}

        new_h = {u: set(neis) for u, neis in adj.items()}
        row_v = new_h[vertex]

        for u, w in combinations(row_v, 2):
            new_h[u].add(w)
            new_h[w].add(u)

        for u in row_v:
            new_h[u].discard(vertex)

        new_h.pop(vertex)

        return {u: sorted(neis) for u, neis in new_h.items()}


    
    '''Returns tree decomposition bags with [bag root: vertices], tree decomposition adjacency with [bag root: neighbor bag roots], root of tree decomposition (root of last bag added)'''
    def dp_tree_decomp(self):
        h_adj = self.g
        
        td_bags: dict[int, list[int]] = {} # key: bag root, value: all vertices in bag
        td_adj: dict[int, list[int]] = {} # key: bag root, value: adjacent bag roots in td
        
        ordering: dict[int, int] = {}
        num_vertices = len(self.g)
        
        root = 0
        count = 0

        for i in range(1, num_vertices+1):
            
            count += 1
            
            min_degree: int = self.min_degree(h=h_adj)
            star_min_deg = self.star(h_adj, min_degree)
            
            td_bags[min_degree] = self.dict_to_list(star_min_deg)
            
            h_adj = self.vertex_elim(h_adj, min_degree)
            ordering[min_degree] = i
            
            if count % 1000 == 0: print(count)
            
        td_adj = {v: [] for v in td_bags}

        for v in self.g:
            # X(v)
            bag = td_bags[v]
            
            count += 1

            # find vertex in X(v) \ {v} with smallest elimination order
            candidates = [u for u in bag if u != v]
            if not candidates:
                continue

            min_u = min(candidates, key=lambda u: ordering[u])
            
            td_adj[v].append(min_u)
            td_adj[min_u].append(v)
            
            root = min_u
        
        for v in self.g:
            td_bags[v].sort(key=lambda x: ordering[x], reverse=True)
                        
        # addl for loop here for reassigning edge weight
        
        return td_bags, td_adj, root

    @staticmethod
    def dict_to_list(adj: dict[int, list[int]]) -> list[int]:
        to_ret = [list(adj.keys())[0]]
        for value in adj.values():
            for v in value:
                to_ret.append(v)

        return to_ret
    
    @staticmethod
    def lca(anc_s, anc_t): # GPT SLOP
        min_len = min(len(anc_s), len(anc_t))
        lca = anc_s[0] 

        for i in range(min_len):
            if anc_s[i] == anc_t[i]:
                lca = anc_s[i]
            else:
                break

        return lca
    
    def h_two_h_query(self, pos, dis, anc, s, t):
        x = self.lca(anc[s], anc[t])
        d = float('inf')
        
        for hierarchy in pos[x]:
            d = min(d, dis[s][hierarchy - 1] + dis[t][hierarchy - 1])
            
        return d 
    
    def h_two_h_naive(self, td_bags, td_adj, root):
        top_down = self.bfs(td_adj, root)[0]
        
        pos: dict[int, list[int]] = {}
        dis: dict[int, list[int]] = {}
        anc: dict[int, list[int]] = {}
        
        count = 0

        for bag_root in top_down:
            
            count += 1
            
            bfs = self.bfs_shortest_path(td_adj, root, bag_root)
            anc_list = bfs[0]
            
            if not bag_root in anc:
                anc[bag_root] = anc_list
            
            vertex_list = td_bags[bag_root]
            
            pos_v = []
            for v in vertex_list:
                pos_v.append(anc[bag_root].index(v) + 1)
                
            pos[bag_root] = sorted(pos_v)
            anc_list = anc[bag_root]
            
            
        for bag_root in top_down:
            count += 1
            
            anc_list = anc[bag_root]
            dis[bag_root] = []
                    
            for item in anc_list:
                dis[bag_root].append(len(self.bfs_shortest_path(self.g, item, bag_root)[0]) - 1)
            
        
        return pos, dis, anc

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
            bfs = self.bfs_shortest_path(td_adj, root, bag_root)
            anc_list = bfs[0]
            
            if not bag_root in anc:
                anc[bag_root] = anc_list


        for bag_root in top_down:
            
            vertex_list = td_bags[bag_root]
            
            pos_v = []
            for v in vertex_list:
                pos_v.append(anc[bag_root].index(v) + 1)
                
            pos[bag_root] = sorted(pos_v)
            anc_list = anc[bag_root]
        
        for bag_root in top_down:
            
            anc_list = anc[bag_root]
            vertex_list = td_bags[bag_root]
            
            if not bag_root in dis:
                dis[bag_root] = []
            
            for i in range(len(anc_list)-1):
                
                dis[bag_root].append(float('inf'))
                
                for j in range(len(vertex_list)-1):
                    if pos[bag_root][j] > i + 1:
                        xj = vertex_list[j]
                        d = dis[xj][i]
                    else:
                        dis_index = anc_list[i]
                        pos_index = pos[bag_root][j] - 1
                        
                        d = dis[dis_index][pos_index]
                    
                    weight = 1
                    dis[bag_root][i] = min(dis[bag_root][i], weight + d)
                    
            dis[bag_root].append(0)

        return pos, dis, anc