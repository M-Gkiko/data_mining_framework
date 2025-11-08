"""Louvain community detection algorithm."""

from typing import List, Dict, Any, Optional, Union
from ...core.community_detection import CommunityDetection
from ...core.network import Network
import networkx as nx
try:
    import community as community_louvain
except ImportError:
    try:
        import community.community_louvain as community_louvain
    except ImportError:
        community_louvain = None


class LouvainCommunityDetection(CommunityDetection):
    """
    Louvain method for community detection.

    Uses modularity optimization to find communities.
    Based on the python-louvain library (community package).
    """

    def __init__(self, resolution: float = 1.0, random_state: Optional[int] = None, **kwargs: Any):
        """
        Initialize Louvain algorithm.

        Args:
            resolution (float): Resolution parameter for modularity (default: 1.0)
            random_state (Optional[int]): Random seed for reproducibility
            **kwargs: Additional parameters
        """
        if community_louvain is None:
            raise ImportError("python-louvain library is required. Install with: pip install python-louvain")

        self.resolution = float(resolution)
        self.random_state = random_state
        self.params = kwargs

        # Results storage
        self._partition: Optional[Dict[Any, int]] = None
        self._modularity: Optional[float] = None

    def fit(self, network: Network, **kwargs: Any) -> None:
        """
        Detect communities using Louvain algorithm.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters
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

        # Run Louvain algorithm
        self._partition = community_louvain.best_partition(
            graph,
            resolution=self.resolution,
            random_state=self.random_state
        )

        # Calculate modularity
        self._modularity = community_louvain.modularity(self._partition, graph)

    def get_communities(self) -> Union[List[List[Any]], Dict[Any, int]]:
        """
        Get the detected communities.

        Returns:
            Dict[Any, int]: Dictionary mapping nodes to community labels
        """
        if self._partition is None:
            return {}
        return self._partition

    def get_modularity(self) -> Optional[float]:
        """
        Get the modularity score.

        Returns:
            Optional[float]: Modularity score
        """
        return self._modularity
