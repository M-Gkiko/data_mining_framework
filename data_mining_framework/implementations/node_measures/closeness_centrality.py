"""Closeness centrality node measure implementation."""

from typing import Dict, Any, Tuple
from ...core.node_measure import NodeMeasure
from ...core.network import Network
from collections import deque


class ClosenessCentralityMeasure(NodeMeasure):
    """
    Closeness centrality measure for nodes.

    Calculates the average distance from each node to all other nodes.
    Higher closeness indicates nodes that can reach others more quickly.
    """

    def __init__(self, normalized: bool = True, **kwargs: Any):
        """
        Initialize Closeness Centrality measure.

        Args:
            normalized (bool): Whether to normalize the scores (default: True)
            **kwargs: Additional parameters
        """
        self.normalized = bool(normalized)
        # keep extra params if provided
        self.params = kwargs

    def calculate(self, network: Network, **kwargs: Any) -> Dict[Any, float]:
        """
        Calculate closeness centrality for all nodes.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters

        Returns:
            Dict[Any, float]: Dictionary mapping nodes to closeness centrality scores
        """
        nodes = list(network.get_nodes())
        n = len(nodes)
        closeness: Dict[Any, float] = {}

        if n == 0:
            return closeness

        # helper BFS to compute shortest path distances from source
        def _bfs_sum_distances(source: Any) -> Tuple[int, float]:
            dist = {source: 0}
            q = deque([source])
            while q:
                u = q.popleft()
                for v in network.get_neighbors(u):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # sum distances to reachable nodes except source
            total = 0
            for node, d in dist.items():
                if node == source:
                    continue
                total += d
            return len(dist), float(total)

        for node in nodes:
            reachable_count, sum_dist = _bfs_sum_distances(node)

            # reachable_count includes the source itself
            if reachable_count <= 1 or sum_dist == 0.0:
                closeness[node] = 0.0
                continue

            # base closeness: (r-1) / sum_dist where r is reachable_count
            r = reachable_count
            base = float(r - 1) / sum_dist

            if self.normalized:
                # normalize by factor (r-1)/(n-1) to account for disconnected graphs
                norm_factor = float(r - 1) / float(max(1, n - 1))
                closeness[node] = base * norm_factor
            else:
                closeness[node] = base

        return closeness
