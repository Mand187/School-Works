from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set
import heapq
from nodeClass import Node

class Graph:
    def __init__(self, directed: bool = True):
        self.node_map: Dict[str, Node] = {}  # Maps node name to Node object
        self.directed = directed  # Whether the graph is directed or undirected

    # Node Operations
    def add_node(self, data: float, name: str) -> bool:
        """Creates node with a data value and a name."""
        if name in self.node_map:
            return False
        self.node_map[name] = Node(data, name)
        return True

    def add_node_default(self, name: str) -> bool:
        """Creates a node with the default data value of '1'."""
        return self.add_node(1.0, name)

    def add_nodes(self, nodes: List[str]):
        """Creates multiple nodes with default data value."""
        for node in nodes:
            self.add_node_default(node)

    def add_nodes_with_data(self, nodes: List[Tuple[float, str]]):
        """Creates multiple nodes with provided data."""
        for data, node in nodes:
            self.add_node(data, node)

    # Edge Operations
    def add_edge(self, from_node: str, to_node: str, weight: float = 1.0) -> bool:
        """Connects from nodes A to B with a given weight."""
        if from_node not in self.node_map or to_node not in self.node_map:
            return False
        self.node_map[from_node].add_neighbor(to_node, weight)
        if not self.directed:
            self.node_map[to_node].add_neighbor(from_node, weight)
        return True

    def delete_node(self, target_node: str) -> bool:
        """Deletes a node from the graph."""
        if target_node not in self.node_map:
            return False
        del self.node_map[target_node]
        for node in self.node_map.values():
            node.get_map().pop(target_node, None)  # Remove edges pointing to this node
        return True

    def delete_edge(self, from_node: str, to_node: str, weight: float = 1.0) -> bool:
        """Deletes an edge between two nodes."""
        if from_node not in self.node_map or to_node not in self.node_map:
            return False
        self.node_map[from_node].get_map()[to_node].discard(weight)
        if not self.directed:
            self.node_map[to_node].get_map()[from_node].discard(weight)
        return True

    # Graph Traversal Operations
    def bfs(self, source_node: str, target_node: str) -> List[str]:
        """Returns the shortest path from source to target."""
        visited = set()
        queue = deque([source_node])
        prev_map = {source_node: None}
        while queue:
            node = queue.popleft()
            if node == target_node:
                break
            for neighbor in self.node_map[node].get_map():
                if neighbor not in visited:
                    visited.add(neighbor)
                    prev_map[neighbor] = node
                    queue.append(neighbor)
        return self._reconstruct_path(prev_map, source_node, target_node)

    def dfs(self, source_node: str, target_node: str) -> List[str]:
        """Returns the shortest path from source to target using DFS."""
        visited = set()
        prev_map = {}
        self._dfs_helper(source_node, target_node, visited, prev_map)
        return self._reconstruct_path(prev_map, source_node, target_node)

    def _dfs_helper(self, node: str, target_node: str, visited: Set[str], prev_map: Dict[str, str]):
        if node == target_node:
            return
        visited.add(node)
        for neighbor in self.node_map[node].get_map():
            if neighbor not in visited:
                prev_map[neighbor] = node
                self._dfs_helper(neighbor, target_node, visited, prev_map)

    def _reconstruct_path(self, prev_map: Dict[str, str], source_node: str, target_node: str) -> List[str]:
        path = []
        node = target_node
        while node is not None:
            path.append(node)
            node = prev_map.get(node)
        return path[::-1] if path[0] == source_node else []

    # Utility functions for getting neighbors
    def neighbor_names(self, name: str) -> List[str]:
        """Returns a list of names of neighboring nodes."""
        return list(self.node_map[name].get_map().keys())

    def neighbor_dist_min(self, name: str) -> List[Tuple[str, float]]:
        """Returns a list of neighbors and their minimum edge weight."""
        return [(neighbor, min(self.node_map[name].get_map()[neighbor])) for neighbor in self.node_map[name].get_map()]

    def neighbor_dist_max(self, name: str) -> List[Tuple[str, float]]:
        """Returns a list of neighbors and their maximum edge weight."""
        return [(neighbor, max(self.node_map[name].get_map()[neighbor])) for neighbor in self.node_map[name].get_map()]

    def delete_neighbors(self, name: str) -> bool:
        """Deletes all neighbors of the specified node."""
        if name not in self.node_map:
            return False
        self.node_map[name].get_map().clear()
        for node in self.node_map.values():
            node.get_map().pop(name, None)  # Remove the reverse edges
        return True

    # Helper functions
    def get_info(self) -> str:
        """Returns a string with all node names and their edges."""
        info = []
        for node_name, node in self.node_map.items():
            info.append(f"{node_name}: {node.get_map()}")
        return "\n".join(info)

    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return len(self.node_map)

    def num_edges(self) -> int:
        """Returns the number of edges in the graph."""
        return sum(len(neighbors) for neighbors in self.node_map.values())

    def node_exists(self, node: str) -> bool:
        """Checks if a node exists in the graph."""
        return node in self.node_map

