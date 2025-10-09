import numpy as np
from typing import List, Optional, Any
from sklearn.cluster import KMeans
from core.clustering import Clustering
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure


class KMeansClustering(Clustering):
    """
    K-Means clustering using scikit-learn, with optional custom distance measure.
    Note: sklearn's KMeans only supports Euclidean distance.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        distance_measure: Optional[DistanceMeasure] = None,
    ):
        """
        Initialize the KMeans clustering object.

        Args:
            n_clusters (int): Number of clusters.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance to declare convergence.
            random_state (Optional[int]): Random seed.
            distance_measure (Optional[DistanceMeasure]): Distance measure (only 'euclidean' is supported).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance_measure = distance_measure

        self._kmeans = None
        self._labels = None
        self._fitted = False

    def fit(self, dataset: Dataset, **kwargs: Any) -> None:
        """
        Fit K-Means clustering to the dataset.

        Args:
            dataset (Dataset): Dataset object containing data
            **kwargs: Optional overrides for n_clusters, max_iter, tol, random_state

        Raises:
            ValueError: If dataset is invalid or custom distance is unsupported
        """
        data = np.asarray(dataset.get_data(), dtype=float)

        if data.ndim != 2:
            raise ValueError("Dataset must be 2D!")

        # Validate distance measure
        if self.distance_measure is not None:
            name = getattr(self.distance_measure, "get_name", lambda: "euclidean")().lower()
            if name != "euclidean":
                raise NotImplementedError(
                    f"KMeans only supports Euclidean distance, but got '{name}'."
                )

        n_clusters = kwargs.get("n_clusters", self.n_clusters)
        max_iter = kwargs.get("max_iter", self.max_iter)
        tol = kwargs.get("tol", self.tol)
        random_state = kwargs.get("random_state", self.random_state)

        self._kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init="auto",
        )

        self._labels = self._kmeans.fit_predict(data).tolist()
        self._fitted = True

    def get_labels(self) -> Optional[List[int]]:
        """Return cluster labels after fitting."""
        return self._labels.copy() if self._fitted else None