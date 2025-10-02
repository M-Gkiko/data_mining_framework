# Hierarchical clustering implementation (Adapter over sklearn)
from typing import List, Optional, Any
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from core.clustering_algorithm import ClusteringAlgorithm
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure


class HierarchicalClustering(ClusteringAlgorithm):
    """
    Adapter for sklearn's AgglomerativeClustering that follows our
    ClusteringAlgorithm interface.

    Notes:
    - To use a custom DistanceMeasure, set metric="precomputed" and we will
      build the full distance matrix using the provided DistanceMeasure.
    - 'ward' linkage requires Euclidean distances and does NOT work with
      'precomputed'. If linkage="ward", we will force metric="euclidean".
    - Newer sklearn uses 'metric' instead of 'affinity'. We support 'metric'
      (preferred) and map 'affinity' -> 'metric' for backward compatibility.
    """

    def __init__(self, **kwargs: Any):
        # Defaults; override via kwargs
        self.params = {
            "n_clusters": 2,
            "linkage": "ward",        # 'ward', 'complete', 'average', 'single'
            "metric": "euclidean",    # 'euclidean', 'manhattan', 'cosine', 'precomputed'
        }
        # Back-compat: allow 'affinity' as alias of 'metric'
        if "affinity" in kwargs and "metric" not in kwargs:
            kwargs = {**kwargs, "metric": kwargs.pop("affinity")}
        self.params.update(kwargs)

        self._labels: Optional[List[int]] = None
        self._model: Optional[AgglomerativeClustering] = None

    def _build_distance_matrix(self, X: np.ndarray, dm: DistanceMeasure) -> np.ndarray:
        n = X.shape[0]
        D = np.zeros((n, n), dtype=float)
        # Compute only upper triangle; mirror to save work
        for i in range(n):
            for j in range(i + 1, n):
                d = dm.calculate(X[i], X[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def fit(self, dataset: Dataset, distance_measure: DistanceMeasure, **kwargs: Any) -> None:
        # Allow runtime overrides
        # Back-compat mapping if someone passes 'affinity'
        if "affinity" in kwargs and "metric" not in kwargs:
            kwargs = {**kwargs, "metric": kwargs.pop("affinity")}
        self.params.update(kwargs)

        X = dataset.get_data()
        if X is None:
            raise ValueError("Dataset.get_data() returned None.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one row.")

        linkage = self.params["linkage"]
        metric = self.params["metric"]

        # Ward requires Euclidean distances and disallows precomputed
        if linkage == "ward":
            if metric == "precomputed":
                raise ValueError("linkage='ward' is incompatible with metric='precomputed'. "
                                 "Use linkage in {'complete','average','single'} for precomputed.")
            metric = "euclidean"  # force compatibility

        # Prepare data or distance matrix
        if metric == "precomputed":
            D = self._build_distance_matrix(X, distance_measure)
            self._model = AgglomerativeClustering(
                n_clusters=self.params["n_clusters"],
                linkage=linkage,
                metric="precomputed",
            )
            self._labels = self._model.fit_predict(D).tolist()
        else:
            # Direct metrics handled by sklearn internally
            self._model = AgglomerativeClustering(
                n_clusters=self.params["n_clusters"],
                linkage=linkage,
                metric=metric,  # sklearn >=1.2
            )
            self._labels = self._model.fit_predict(X).tolist()

    def get_labels(self) -> Optional[List[int]]:
        return self._labels
