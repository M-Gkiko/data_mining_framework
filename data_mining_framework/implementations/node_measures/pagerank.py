"""PageRank node measure implementation."""

from typing import Dict, Any
from ...core.node_measure import NodeMeasure
from ...core.network import Network
import networkx as nx


class PageRankMeasure(NodeMeasure):
    """
    PageRank centrality measure for nodes.

    Calculates the importance of nodes based on the link structure.
    Higher PageRank indicates more important/influential nodes.
    """

    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6, **kwargs: Any):
        """
        Initialize PageRank measure.

        Args:
            alpha (float): Damping parameter (default: 0.85)
            max_iter (int): Maximum number of iterations (default: 100)
            tol (float): Error tolerance for convergence (default: 1e-6)
            **kwargs: Additional parameters
        """
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.params = kwargs

    def calculate(self, network: Network, **kwargs: Any) -> Dict[Any, float]:
        """
        Calculate PageRank for all nodes.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters

        Returns:
            Dict[Any, float]: Dictionary mapping nodes to PageRank scores
        """
        # Convert to NetworkX graph if needed
        if hasattr(network, 'get_networkx_graph'):
            # Network is already a NetworkXWrapper
            graph = network.get_networkx_graph()
        else:
            # Build a NetworkX graph from the network interface
            graph = nx.Graph()
            graph.add_nodes_from(network.get_nodes())
            graph.add_edges_from(network.get_edges())

        # Calculate PageRank using NetworkX
        pagerank_scores = nx.pagerank(
            graph,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol
        )

        return pagerank_scores
