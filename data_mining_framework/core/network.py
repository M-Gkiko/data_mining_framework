from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Set


class Network(ABC):
    """
    Abstract base class for network/graph data structures.

    This interface defines the contract that all network implementations
    must follow, enabling the Strategy Pattern for network handling.
    """

    @abstractmethod
    def get_nodes(self) -> List[Any]:
        """
        Get all nodes in the network.

        Returns:
            List[Any]: List of all node identifiers
        """
        pass

    @abstractmethod
    def get_edges(self) -> List[Tuple[Any, Any]]:
        """
        Get all edges in the network.

        Returns:
            List[Tuple[Any, Any]]: List of edges as (source, target) tuples
        """
        pass

    @abstractmethod
    def get_neighbors(self, node: Any) -> List[Any]:
        """
        Get all neighbors of a given node.

        Args:
            node (Any): The node identifier

        Returns:
            List[Any]: List of neighbor node identifiers
        """
        pass

    @abstractmethod
    def node_count(self) -> int:
        """
        Get the total number of nodes in the network.

        Returns:
            int: Number of nodes
        """
        pass

    @abstractmethod
    def edge_count(self) -> int:
        """
        Get the total number of edges in the network.

        Returns:
            int: Number of edges
        """
        pass

    @abstractmethod
    def has_edge(self, source: Any, target: Any) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            source (Any): Source node identifier
            target (Any): Target node identifier

        Returns:
            bool: True if edge exists, False otherwise
        """
        pass

    @abstractmethod
    def get_adjacency_matrix(self):
        """
        Get the adjacency matrix representation of the network.

        Returns:
            Matrix representation (implementation-specific type)
        """
        pass
