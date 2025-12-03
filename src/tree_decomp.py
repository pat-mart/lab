from typing import TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    from graph import Graph


class TreeDecomp:
    """
    Tree decomposition class where bags are Graph instances.
    Uses adjacency dict to represent the tree structure.
    """

    def __init__(self):
        # Adjacency dict: Graph -> list of neighbor Graphs
        # Using 'any' at runtime to avoid circular import
        self.adj: dict = {}
        # Root of the tree decomposition
        self.root: Graph | None = None
        # Track all bags in the decomposition
        self.bags: set = set()

    def add_bag(self, bag, root):
        """
        Add a bag to the tree decomposition.

        Args:
            bag: Either a Graph instance or adjacency dict to create a Graph from
            root: The v value from which this bag X(v) is created
        """
        if isinstance(bag, dict):
            from src.graph import Graph
            bag = Graph(adjacency_dict=bag)
            bag.root = root

        bag.root = root
        self.bags.add(bag)
        self.adj[bag] = []

    def add_edge(self, u, v):
        """
        Creates bidirectional edge from u to v, where u and v are bags.
        Graphs are identified by their root vertex - if a graph with the same
        root already exists in the tree, that existing graph is used instead.

        Args:
            u: First bag (Graph instance)
            v: Second bag (Graph instance)
        """
        # Find existing bags with matching roots
        u_existing = self.find_bag(u)
        v_existing = self.find_bag(v)
        
        if self.root is None:
            self.root = v_existing

        # Add bidirectional edge
        if v_existing not in self.adj[u_existing]:
            self.adj[u_existing].append(v_existing)
        if u_existing not in self.adj[v_existing]:
            self.adj[v_existing].append(u_existing)

    def find_bag(self, bag):
        """
        Find an existing bag with the same root, or add this bag if none exists.
        
        Args:
            bag: Graph instance to find or add
            
        Returns:
            The existing bag with matching root, or the provided bag if none exists
        """
        # Check if a bag with this root already exists
        for existing_bag in self.adj.keys():
            if existing_bag.root == bag.root:
                return existing_bag

    def get_neighbors(self, bag):
        """Get all neighbor bags of a given bag."""
        return self.adj.get(bag)

    def get_root(self):
        """Get the root of the tree decomposition."""
        return self.root

    def num_bags(self):
        """Get the number of bags in the decomposition."""
        return len(self.bags)

    def get_bag_from_root(self, graph_root: int):
        for bag in self.bags:
            if bag.root == graph_root:
                return bag

        return None

    def anc(self, bag) -> list[int]:
        
        if self.root is bag:
            return [self.root.root]

        stack = [(self.root, [self.root.root])]
        seen = {self.root}
        

        while stack:
            node, path = stack.pop()

            for neighbor in self.get_neighbors(node):
                if neighbor not in seen:
                    seen.add(neighbor)
                    new_path = path + [neighbor.root]

                    if neighbor is bag:
                        return sorted(new_path)

                    stack.append((neighbor, new_path))

        return sorted(path)
    
    # gets distance array, or the distance of the root vertex of xv or X(v)
    # to every node in xv.anc
    def dis(self, xv, xv_anc, g) -> list[int]:
        v = xv.root
        
        dis_list = []
        
        for node in xv_anc:
            bfs_result = g.bfs(start_key=v, goal_key=node)
            dis_list.append(len(bfs_result))
        
        return dis_list
    
    
    def top_down(self):
        if self.root is None:
            return []
    
        order = []
        queue = deque([self.root])
        visited = set([self.root])
        
        while queue:
            current = queue.popleft()
            order.append(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return order
        

    def tree_width(self):
        """
        Calculate the tree width of the decomposition.
        Tree width = max(|bag| - 1) over all bags.
        """
        if not self.bags:
            return -1

        max_bag_size = 0
        for bag in self.bags:
            bag_size = len(list(bag.g.values())[0]) + 1 # denotes values in each bag, and the vertex they are adjacent to (hence +1)
            max_bag_size = max(max_bag_size, bag_size)

        return max_bag_size - 1

    def __str__(self):
        """String representation of the tree decomposition."""
        lines = [f"TreeDecomp with {self.num_bags()} bags"]
        if self.root:
            lines.append(f"Root bag has {len(self.root.g)} vertices")
        lines.append(f"Treewidth: {self.tree_width()}")
        return "\n".join(lines)