"""Jaccard coefficient edge measure implementation."""

from typing import Dict, Tuple, Any
from ...core.edge_measure import EdgeMeasure
from ...core.network import Network


class JaccardCoefficientMeasure(EdgeMeasure):
    """
    Jaccard coefficient measure for edges.

    Calculates the Jaccard similarity between the neighbor sets of two nodes.
    Higher coefficient indicates greater similarity of neighborhoods.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize Jaccard Coefficient measure.

        Args:
            **kwargs: Additional parameters
        """
        # no special parameters needed currently; keep for compatibility
        # Accept and ignore any kwargs to be flexible with registry
        self.params = kwargs

    def calculate(self, network: Network, **kwargs: Any) -> Dict[Tuple[Any, Any], float]:
        """
        Calculate Jaccard coefficient for all edges.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters

        Returns:
            Dict[Tuple[Any, Any], float]: Dictionary mapping edges to Jaccard coefficients
        """
        scores: Dict[Tuple[Any, Any], float] = {}

        # iterate over edges reported by the network implementation
        for edge in network.get_edges():
            u, v = edge

            # neighbor sets (as sets for efficient ops)
            neigh_u = set(network.get_neighbors(u))
            neigh_v = set(network.get_neighbors(v))

            # Jaccard on neighbor sets (exclude the nodes themselves if present)
            if u in neigh_u:
                neigh_u.discard(u)
            if v in neigh_v:
                neigh_v.discard(v)

            inter = neigh_u & neigh_v
            union = neigh_u | neigh_v

            if not union:
                score = 0.0
            else:
                score = float(len(inter)) / float(len(union))

            scores[(u, v)] = score

        return scores

if __name__ == "__main__":
    import os, pprint
    from ...implementations.networks.edgelist import EdgeListNetwork

    # Path to your example edgelist
    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, "../networks/example_edgelist.txt")

    # Load the network
    net = EdgeListNetwork(filepath, directed=False)

    # Initialize and calculate the measure
    measure = JaccardCoefficientMeasure()
    results = measure.calculate(net)

    print("Jaccard Coefficient Results:")
    pprint.pprint(results)


