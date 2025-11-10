"""Girvan-Newman community detection algorithm."""

from typing import List, Dict, Any, Optional, Union
import networkx as nx
from ...core.community_detection import CommunityDetection
from ...core.network import Network
from ..edge_measures.betweenness import EdgeBetweennessMeasure


class GirvanNewmanCommunity(CommunityDetection):
    """
    Girvan-Newman algorithm for community detection.

    Hierarchical method that iteratively removes edges with highest betweenness.
    Based on NetworkX's girvan_newman algorithm.
    """

    def __init__(self, k: int = 2, **kwargs: Any):
        """
        Initialize Girvan-Newman algorithm.

        Args:
            k (int): Number of communities to detect (default: 2)
            **kwargs: Additional parameters
        """
        self.k = k
        self._communities: List[List[Any]] = []
        self._modularity: Optional[float] = None
        self._fitted = False

    def fit(self, network: Network, **kwargs: Any) -> None:
        """
        Detect communities using Girvan-Newman algorithm.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters

        Raises:
            ValueError: If network is empty or has fewer nodes than k
        """
        if network.node_count() == 0:
            raise ValueError("Network must have at least one node")

        if network.node_count() < self.k:
            raise ValueError(f"Network has {network.node_count()} nodes but k={self.k} communities requested")

        G = nx.Graph()
        G.add_nodes_from(network.get_nodes())
        G.add_edges_from(network.get_edges())

        communities_generator = nx.community.girvan_newman(G)

        communities = None
        for _ in range(self.k):
            try:
                communities = next(communities_generator)
            except StopIteration:
                break

        if communities is None:
            communities = list(G.nodes())

        self._communities = [list(community) for community in communities]

        self._modularity = nx.community.modularity(G, self._communities)
        self._fitted = True

    def get_communities(self) -> Union[List[List[Any]], Dict[Any, int]]:
        """
        Get the detected communities.

        Returns:
            List[List[Any]]: List of communities (each community is a list of nodes)

        Raises:
            ValueError: If fit() has not been called
        """
        if not self._fitted:
            raise ValueError("Must call fit() before getting communities")

        return [community.copy() for community in self._communities]

    def get_modularity(self) -> Optional[float]:
        """
        Get the modularity score.

        Returns:
            Optional[float]: Modularity score, or None if fit() not called

        Raises:
            ValueError: If fit() has not been called
        """
        if not self._fitted:
            raise ValueError("Must call fit() before getting modularity")

        return self._modularity
