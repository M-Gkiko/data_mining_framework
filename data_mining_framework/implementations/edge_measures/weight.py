"""Edge weight measure implementation."""

from typing import Dict, Tuple, Any
from ...core.edge_measure import EdgeMeasure
from ...core.network import Network
import networkx as nx


class EdgeWeightMeasure(EdgeMeasure):
    """
    Simple edge weight measure.

    Extracts the weight of each edge from a weighted network.
    For unweighted networks, returns 1.0 for all edges.
    """

    def __init__(self, weight_attribute: str = 'weight', default_weight: float = 1.0, **kwargs: Any):
        """
        Initialize Edge Weight measure.

        Args:
            weight_attribute (str): Name of the weight attribute (default: 'weight')
            default_weight (float): Default weight for unweighted edges (default: 1.0)
            **kwargs: Additional parameters
        """
        self.weight_attribute = str(weight_attribute)
        self.default_weight = float(default_weight)
        self.params = kwargs

    def calculate(self, network: Network, **kwargs: Any) -> Dict[Tuple[Any, Any], float]:
        """
        Calculate edge weights for all edges.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters

        Returns:
            Dict[Tuple[Any, Any], float]: Dictionary mapping edges to weights
        """
        weights: Dict[Tuple[Any, Any], float] = {}

        # Check if network is a NetworkXWrapper with edge attributes
        if hasattr(network, 'get_networkx_graph'):
            graph = network.get_networkx_graph()
            # Extract weights from NetworkX graph edge attributes
            for u, v in network.get_edges():
                edge_data = graph.get_edge_data(u, v)
                if edge_data and self.weight_attribute in edge_data:
                    weights[(u, v)] = float(edge_data[self.weight_attribute])
                else:
                    weights[(u, v)] = self.default_weight
        else:
            # For generic Network implementations, return default weight for all edges
            for edge in network.get_edges():
                weights[edge] = self.default_weight

        return weights
