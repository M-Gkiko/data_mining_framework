"""Label propagation community detection algorithm."""

from typing import List, Dict, Any, Optional, Union
from ...core.community_detection import CommunityDetection
from ...core.network import Network
import random
from collections import Counter


class LabelPropagationCommunity(CommunityDetection):
    """
    Label Propagation algorithm for community detection.

    Nodes iteratively adopt the label that most of their neighbors have.
    Based on NetworkX's label_propagation_communities.
    """

    def __init__(self, max_iter: int = 100, seed: Optional[int] = None, **kwargs: Any):
        """
        Initialize Label Propagation algorithm.

        Args:
            max_iter (int): Maximum number of iterations (default: 100)
            seed (Optional[int]): Random seed for reproducibility
            **kwargs: Additional parameters
        """
        self.max_iter = int(max_iter)
        self.seed = seed
        # random generator (if seed is None, will be system-random)
        self._rng = random.Random(seed)

        # results
        self._labels: Dict[Any, Any] = {}
        self._communities: Optional[Dict[Any, List[Any]]] = None
        self._modularity: Optional[float] = None
        # store other params if provided
        self.params = kwargs

    def fit(self, network: Network, **kwargs: Any) -> None:
        """
        Detect communities using Label Propagation.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional parameters
        """
        nodes = list(network.get_nodes())
        # initialize labels (unique initial label per node)
        labels: Dict[Any, Any] = {n: n for n in nodes}

        for it in range(self.max_iter):
            changed = False

            # process nodes in random order each iteration
            self._rng.shuffle(nodes)

            for node in nodes:
                neigh = network.get_neighbors(node)
                if not neigh:
                    continue

                # collect neighbor labels
                neigh_labels = [labels[n] for n in neigh if n in labels]
                if not neigh_labels:
                    continue

                counts = Counter(neigh_labels)
                max_count = max(counts.values())
                top_labels = [lab for lab, cnt in counts.items() if cnt == max_count]

                # break ties randomly but deterministically via RNG
                chosen = self._rng.choice(top_labels)

                if labels.get(node) != chosen:
                    labels[node] = chosen
                    changed = True

            if not changed:
                break

        # store labels
        self._labels = labels

        # build communities mapping label -> list(nodes)
        comms: Dict[Any, List[Any]] = {}
        for node, lab in labels.items():
            comms.setdefault(lab, []).append(node)

        # store as dict and also as list-of-lists when requested
        self._communities = comms

        # compute modularity if possible
        try:
            self._modularity = self._compute_modularity(network, comms)
        except Exception:
            self._modularity = None

    def get_communities(self) -> Union[List[List[Any]], Dict[Any, int]]:
        """
        Get the detected communities.

        Returns:
            List[List[Any]]: List of communities (each community is a list of nodes)
        """
        if self._communities is None:
            return []
        # return list of communities (lists of nodes)
        return list(self._communities.values())

    def get_modularity(self) -> Optional[float]:
        """
        Get the modularity score.

        Returns:
            Optional[float]: Modularity score, or None
        """
        return self._modularity

    def _compute_modularity(self, network: Network, comms: Dict[Any, List[Any]]) -> Optional[float]:
        """
        Compute modularity for the given partition. Uses standard undirected modularity
        Q = (1/2m) sum_{ij} [A_ij - k_i k_j / (2m)] delta(c_i, c_j)
        Returns None if modularity cannot be computed (e.g., zero edges).
        """
        m = network.edge_count()
        if m == 0:
            return None

        # degree for each node (use neighbor count)
        degrees: Dict[Any, int] = {n: len(network.get_neighbors(n)) for n in network.get_nodes()}

        two_m = 2.0 * float(m)
        q = 0.0

        # iterate communities; sum over node pairs in each community
        for lab, nodes in comms.items():
            for u in nodes:
                for v in nodes:
                    a_uv = 1.0 if network.has_edge(u, v) else 0.0
                    k_u = float(degrees.get(u, 0))
                    k_v = float(degrees.get(v, 0))
                    q += (a_uv - (k_u * k_v) / two_m)

        q = q / two_m
        return q
