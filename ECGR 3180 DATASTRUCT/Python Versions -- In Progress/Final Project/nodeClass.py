from collections import defaultdict
from typing import DefaultDict, Set

class Node:
    def __init__(self, data: float, name: str):
        """
        Initializes the node with provided data and name.
        """
        self.data = data
        self.name = name
        # Use defaultdict of sets to store neighbors and their edge weights
        self.neighbor_map: DefaultDict[str, Set[float]] = defaultdict(set)
        # Set to track neighbors of this node (if needed for other purposes)
        self.neighbor_of_set: Set[str] = set()

    def add_neighbor(self, neighbor_name: str, weight: float):
        """
        Adds a neighbor with a given name and edge weight.
        """
        # Add the weight to the set of weights for the neighbor
        self.neighbor_map[neighbor_name].add(weight)

    def get_data(self) -> float:
        """
        Returns the data associated with the node.
        """
        return self.data

    def get_map(self) -> DefaultDict[str, Set[float]]:
        """
        Returns a reference to the neighbor map.
        """
        return self.neighbor_map

    def get_neighbor_of_set(self) -> Set[str]:
        """
        Returns a reference to the neighbor_of_set.
        """
        return self.neighbor_of_set
