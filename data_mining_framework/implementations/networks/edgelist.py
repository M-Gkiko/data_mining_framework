"""Edge list network implementation."""

from typing import Any, List, Tuple, Dict, Set, Optional
from ...core.network import Network


class EdgeListNetwork(Network):
    """
    Network implementation that reads from edge list files.

    Edge list format: Each line contains two node identifiers (source, target)
    separated by whitespace or comma.
    """

    def __init__(self, filepath: str, delimiter: Optional[str] = None, directed: bool = False):
        """
        Initialize EdgeListNetwork from a file.

        Args:
            filepath (str): Path to the edge list file
            delimiter (Optional[str]): Delimiter between node IDs (default: whitespace)
            directed (bool): Whether the network is directed
        """
        self._directed = directed
        self._nodes: Set[Any] = set()
        self._edges: List[Tuple[Any, Any]] = []
        self._edge_set: Set[Tuple[Any, Any]] = set()
        self._adj: Dict[Any, Set[Any]] = {}

        # Read file and populate structures
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # split by provided delimiter or by comma/whitespace
                if delimiter:
                    parts = [p for p in line.split(delimiter) if p != ""]
                else:
                    # allow comma or whitespace as separators
                    parts = [p for p in line.replace(",", " ").split() if p != ""]
                if len(parts) < 2:
                    continue
                src, tgt = parts[0], parts[1]

                # record nodes
                self._nodes.add(src)
                self._nodes.add(tgt)

                # record edge and adjacency
                self._edges.append((src, tgt))
                self._edge_set.add((src, tgt))
                self._adj.setdefault(src, set()).add(tgt)
                if not self._directed:
                    self._adj.setdefault(tgt, set()).add(src)

    def get_nodes(self) -> List[Any]:
        """Get all nodes in the network."""
        return list(self._nodes)

    def get_edges(self) -> List[Tuple[Any, Any]]:
        """Get all edges in the network."""
        return list(self._edges)

    def get_neighbors(self, node: Any) -> List[Any]:
        """Get all neighbors of a given node."""
        return list(self._adj.get(node, []))

    def node_count(self) -> int:
        """Get the total number of nodes."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Get the total number of edges."""
        return len(self._edges)

    def has_edge(self, source: Any, target: Any) -> bool:
        """Check if an edge exists."""
        if (source, target) in self._edge_set:
            return True
        if not self._directed and (target, source) in self._edge_set:
            return True
        return False

    def get_adjacency_matrix(self) -> Dict[Any, Dict[Any, int]]:
        """Get adjacency matrix representation as nested dicts with 1/0 values."""
        nodes = list(self._nodes)
        matrix: Dict[Any, Dict[Any, int]] = {}
        for n in nodes:
            row: Dict[Any, int] = {}
            neigh = self._adj.get(n, set())
            for m in nodes:
                row[m] = 1 if m in neigh else 0
            matrix[n] = row
        return matrix
