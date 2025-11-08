"""NetworkX wrapper implementation."""

from typing import Any, List, Tuple, Dict, Set
import networkx as nx
from ...core.network import Network


class NetworkXWrapper(Network):
    """
    Network implementation that wraps NetworkX graph objects.

    This allows using NetworkX's rich functionality while conforming
    to the framework's Network interface.
    """

    def __init__(self, graph: nx.Graph = None, filepath: str = None, format: str = 'edgelist'):
        """
        Initialize NetworkXWrapper.

        Args:
            graph (nx.Graph): Existing NetworkX graph
            filepath (str): Path to network file (if graph is None)
            format (str): Format of the file ('edgelist', 'gml', 'graphml', etc.)
        """
        if graph is not None:
            self._graph = graph
        elif filepath is not None:
            # Load graph from file based on format
            if format == 'edgelist':
                self._graph = nx.read_edgelist(filepath)
            elif format == 'gml':
                self._graph = nx.read_gml(filepath)
            elif format == 'graphml':
                self._graph = nx.read_graphml(filepath)
            elif format == 'gexf':
                self._graph = nx.read_gexf(filepath)
            elif format == 'adjlist':
                self._graph = nx.read_adjlist(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            # Create empty graph
            self._graph = nx.Graph()

    def get_nodes(self) -> List[Any]:
        """Get all nodes in the network."""
        return list(self._graph.nodes())

    def get_edges(self) -> List[Tuple[Any, Any]]:
        """Get all edges in the network."""
        return list(self._graph.edges())

    def get_neighbors(self, node: Any) -> List[Any]:
        """Get all neighbors of a given node."""
        return list(self._graph.neighbors(node))

    def node_count(self) -> int:
        """Get the total number of nodes."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Get the total number of edges."""
        return self._graph.number_of_edges()

    def has_edge(self, source: Any, target: Any) -> bool:
        """Check if an edge exists."""
        return self._graph.has_edge(source, target)

    def get_adjacency_matrix(self):
        """Get adjacency matrix representation."""
        return nx.to_numpy_array(self._graph)

    def get_networkx_graph(self) -> nx.Graph:
        """
        Get the underlying NetworkX graph object.

        Returns:
            nx.Graph: The NetworkX graph
        """
        return self._graph
