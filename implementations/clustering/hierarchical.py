# Hierarchical clustering implementation (Adapter over sklearn)
from typing import List, Optional, Any
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from core.clustering import Clustering
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure
from utils.distance_utils import build_distance_matrix


class HierarchicalClustering(Clustering):
    """
    Adapter for sklearn's AgglomerativeClustering that follows our
    ClusteringAlgorithm interface.

    Defaults to metric='precomputed' to encourage use of custom DistanceMeasures.

    Notes:
    - Default behavior uses metric='precomputed' with custom distance measures
    - 'ward' linkage is incompatible with 'precomputed' metric - explicit error thrown
    - To use sklearn built-in metrics, explicitly set metric (e.g., 'euclidean') 
      and do NOT provide distance_measure parameter
    - Newer sklearn uses 'metric' instead of 'affinity'. We support 'metric'
      (preferred) and map 'affinity' -> 'metric' for backward compatibility.
    """

    def __init__(self, **kwargs: Any):
        # Defaults; override via kwargs
        self.params = {
            "n_clusters": 2,
            "linkage": "complete",    # 'complete', 'average', 'single' (ward not compatible with precomputed)
            "metric": "precomputed",  # Default to precomputed to use custom distance measures
        }
        # Back-compat: allow 'affinity' as alias of 'metric'
        if "affinity" in kwargs and "metric" not in kwargs:
            kwargs = {**kwargs, "metric": kwargs.pop("affinity")}
        self.params.update(kwargs)

        self._labels: Optional[List[int]] = None
        self._model: Optional[AgglomerativeClustering] = None

    def fit(self, dataset: Dataset, distance_measure: Optional[DistanceMeasure] = None, **kwargs: Any) -> None:
        """
        Fit the hierarchical clustering algorithm to the given dataset.
        
        Args:
            dataset (Dataset): The dataset to cluster
            distance_measure (Optional[DistanceMeasure]): Custom distance measure (required for default metric='precomputed')
            **kwargs: Optional hyperparameters including:
                - n_clusters (int): Number of clusters to find (default: 2)
                - linkage (str): Linkage criterion ('complete', 'average', 'single') (default: 'complete')
                - metric (str): Distance metric ('precomputed', 'euclidean', 'manhattan', 'cosine') (default: 'precomputed')
                
        Raises:
            ValueError: If dataset is invalid or incompatible parameters are provided
                       (e.g., linkage='ward' with metric='precomputed', or missing distance_measure with metric='precomputed')
        """
        
        # Allow runtime overrides
        # Back-compat mapping if someone passes 'affinity'
        if "affinity" in kwargs and "metric" not in kwargs:
            kwargs = {**kwargs, "metric": kwargs.pop("affinity")}
        
        # Remove distance_measure from kwargs before updating params
        fit_kwargs = {k: v for k, v in kwargs.items() if k != 'distance_measure'}
        self.params.update(fit_kwargs)

        X = dataset.get_data()
        if X is None:
            raise ValueError("Dataset.get_data() returned None.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one row.")

        linkage = self.params["linkage"]
        metric = self.params["metric"]

        # Explicit error checking for incompatible combinations
        if linkage == "ward" and metric == "precomputed":
            raise ValueError(
                "linkage='ward' is incompatible with metric='precomputed'. "
                "Either use linkage in {'complete', 'average', 'single'} with metric='precomputed', "
                "or use linkage='ward' with metric='euclidean'"
            )

        if metric != "precomputed" and distance_measure is not None:
            raise ValueError(
                f"distance_measure provided but metric='{metric}'. "
                "To use custom distance measures, set metric='precomputed'"
            )

        # Prepare data or distance matrix
        if metric == "precomputed":
            if distance_measure is None:
                raise ValueError(
                    "metric='precomputed' requires a distance_measure parameter. "
                    "Either provide distance_measure or use a built-in metric like 'euclidean'"
                )
            D = build_distance_matrix(X, distance_measure)
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
