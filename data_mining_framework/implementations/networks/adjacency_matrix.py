"""Adjacency matrix network implementation."""

from typing import Any, List, Tuple, Dict, Set
import numpy as np
from ...core.network import Network


class AdjacencyMatrixNetwork(Network):
    """
    Network implementation that reads from adjacency matrix files.

    Adjacency matrix format: N x N matrix where entry (i,j) indicates
    connection between node i and node j.
    """

    def __init__(self, filepath: str = None, matrix: np.ndarray = None, directed: bool = False):
        """
        Initialize AdjacencyMatrixNetwork.

        Args:
            filepath (str): Path to adjacency matrix file (CSV format)
            matrix (np.ndarray): Pre-loaded adjacency matrix
            directed (bool): Whether the network is directed

        Raises:
            ValueError: If neither filepath nor matrix is provided, or if invalid format
        """
        if filepath is None and matrix is None:
            raise ValueError("Either filepath or matrix must be provided")

        if filepath is not None:
            self._matrix = np.loadtxt(filepath, delimiter=',')
        else:
            self._matrix = np.asarray(matrix, dtype=float)

        if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square 2D array")

        self._directed = directed
        self._nodes = list(range(self._matrix.shape[0]))
        self._edge_cache: Dict[Tuple[Any, Any], bool] = {}

    def get_nodes(self) -> List[Any]:
        """Get all nodes in the network."""
        return self._nodes.copy()

    def get_edges(self) -> List[Tuple[Any, Any]]:
        """Get all edges in the network."""
        edges = []
        n = len(self._nodes)

        if self._directed:
            for i in range(n):
                for j in range(n):
                    if self._matrix[i, j] != 0:
                        edges.append((self._nodes[i], self._nodes[j]))
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    if self._matrix[i, j] != 0 or self._matrix[j, i] != 0:
                        edges.append((self._nodes[i], self._nodes[j]))

        return edges

    def get_neighbors(self, node: Any) -> List[Any]:
        """Get all neighbors of a given node."""
        if node not in self._nodes:
            raise ValueError(f"Node {node} not in network")

        idx = self._nodes.index(node)
        neighbors = []
        n = len(self._nodes)

        if self._directed:
            for j in range(n):
                if self._matrix[idx, j] != 0:
                    neighbors.append(self._nodes[j])
            for i in range(n):
                if self._matrix[i, idx] != 0 and self._nodes[i] != node:
                    if self._nodes[i] not in neighbors:
                        neighbors.append(self._nodes[i])
        else:
            for j in range(n):
                if j != idx and (self._matrix[idx, j] != 0 or self._matrix[j, idx] != 0):
                    neighbors.append(self._nodes[j])

        return neighbors

    def node_count(self) -> int:
        """Get the total number of nodes."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Get the total number of edges."""
        return len(self.get_edges())

    def has_edge(self, source: Any, target: Any) -> bool:
        """Check if an edge exists."""
        if source not in self._nodes or target not in self._nodes:
            return False

        idx_i = self._nodes.index(source)
        idx_j = self._nodes.index(target)

        if self._directed:
            return self._matrix[idx_i, idx_j] != 0
        else:
            return self._matrix[idx_i, idx_j] != 0 or self._matrix[idx_j, idx_i] != 0

    def get_adjacency_matrix(self):
        """Get adjacency matrix representation."""
        return self._matrix.copy()
