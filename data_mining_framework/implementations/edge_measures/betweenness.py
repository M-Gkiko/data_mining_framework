"""Edge betweenness measure implementation."""

from typing import Dict, Tuple, Any, List
from collections import defaultdict, deque
from ...core.edge_measure import EdgeMeasure
from ...core.network import Network


class EdgeBetweennessMeasure(EdgeMeasure):
    """
    Edge betweenness centrality measure.

    Calculates the number of shortest paths that pass through each edge.
    Higher betweenness indicates edges that are more critical for network connectivity.
    """

    def __init__(self, normalized: bool = True, **kwargs: Any):
        """
        Initialize Edge Betweenness measure.

        Args:
            normalized (bool): Whether to normalize the scores (default: True)
            **kwargs: Additional parameters
        """
        self.normalized = normalized
        self._scores: Dict[Tuple[Any, Any], float] = {}

    def calculate(self, network: Network, **kwargs: Any) -> Dict[Tuple[Any, Any], float]:
        """
        Calculate edge betweenness for all edges.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters

        Returns:
            Dict[Tuple[Any, Any], float]: Dictionary mapping edges to betweenness scores

        Raises:
            ValueError: If network is empty
        """
        if network.node_count() == 0:
            raise ValueError("Network must have at least one node")

        self._scores = defaultdict(float)
        nodes = network.get_nodes()
        n = len(nodes)

        for source in nodes:
            stack = []
            paths = {node: [] for node in nodes}
            sigma = defaultdict(float)  # Number of shortest paths
            distance = {node: -1 for node in nodes}

            sigma[source] = 1.0
            distance[source] = 0
            queue = deque([source])

            while queue:
                v = queue.popleft()
                stack.append(v)

                for w in network.get_neighbors(v):
                    if distance[w] < 0:
                        distance[w] = distance[v] + 1
                        queue.append(w)

                    if distance[w] == distance[v] + 1:
                        sigma[w] += sigma[v]
                        paths[w].append(v)

            edge_delta = defaultdict(float)

            while stack:
                w = stack.pop()

                for v in paths[w]:
                    coeff = (sigma[v] / sigma[w]) * (1.0 + edge_delta[(w, v)] if (w, v) in edge_delta else 1.0)
                    edge_delta[(w, v)] += coeff

                    edge = tuple(sorted([v, w]))
                    self._scores[edge] += coeff

        if not kwargs.get("directed", False):
            for edge in self._scores:
                self._scores[edge] /= 2.0

        if self.normalized and n > 1:
            norm_factor = 2.0 / (n * (n - 1))
            for edge in self._scores:
                self._scores[edge] *= norm_factor

        return dict(self._scores)
