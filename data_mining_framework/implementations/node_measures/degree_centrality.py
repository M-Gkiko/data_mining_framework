"""Degree centrality node measure implementation."""

from typing import Dict, Any
from ...core.node_measure import NodeMeasure
from ...core.network import Network


class DegreeCentralityMeasure(NodeMeasure):
    """
    Degree centrality measure for nodes.

    Calculates the number of connections each node has.
    Higher degree indicates more connected nodes.
    """

    def __init__(self, normalized: bool = True, **kwargs: Any):
        """
        Initialize Degree Centrality measure.

        Args:
            normalized (bool): Whether to normalize by max possible degree (default: True)
            **kwargs: Additional parameters
        """
        self.normalized = normalized
        self._scores: Dict[Any, float] = {}

    def calculate(self, network: Network, **kwargs: Any) -> Dict[Any, float]:
        """
        Calculate degree centrality for all nodes.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters

        Returns:
            Dict[Any, float]: Dictionary mapping nodes to degree centrality scores

        Raises:
            ValueError: If network is empty
        """
        if network.node_count() == 0:
            raise ValueError("Network must have at least one node")

        self._scores = {}
        n = network.node_count()
        divisor = (n - 1) if self.normalized and n > 1 else 1

        for node in network.get_nodes():
            degree = len(network.get_neighbors(node))
            self._scores[node] = degree / divisor if divisor > 0 else 0.0

        return self._scores.copy()
