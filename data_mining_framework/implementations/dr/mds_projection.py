from typing import Any, Optional
import numpy as np
from sklearn.manifold import MDS
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure
from ...core.dimensionality_reduction import DimensionalityReduction
from ...utils.distance_utils import build_distance_matrix


class MDSProjection(DimensionalityReduction):
    """
    Adapter for sklearn's Multidimensional Scaling (MDS)
    restricted to custom DistanceMeasure implementations.

    - Uses both DistanceMeasure or sklearn's built-in metrics.
    - Returns a 2D numpy array (rows = samples, columns = components).
    """

    def __init__(
        self,
        distance_measure: Optional[DistanceMeasure] = None,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        max_iter: int = 300,
        n_init: int = 4,
        **kwargs: Any,
    ):
        """Minimal constructor exposing the essential MDS parameters.

        Extra kwargs may be provided but will only be used to override the
        internal params dictionary.
        """
        # Store distance measure (may be None)
        self.distance_measure = distance_measure

        # Minimal explicit parameters
        # Note: sklearn MDS 'metric' parameter is boolean (True for metric MDS, False for non-metric)
        # When using custom distance measures, we use metric=True with dissimilarity='precomputed'
        if distance_measure is not None:
            # Use metric MDS with precomputed distances
            default_metric = True
            default_dissimilarity = "precomputed"
        else:
            # Use sklearn's built-in distance computation
            default_metric = kwargs.pop("metric", True)
            default_dissimilarity = kwargs.pop("dissimilarity", "euclidean")

        if "metric" in kwargs and distance_measure is not None:
            raise ValueError(
                "Cannot override metric when using custom distance_measure. "
                "Custom distance measures always use metric=True with dissimilarity='precomputed'"
            )

        self.params = {
            "n_components": int(n_components),
            "random_state": int(random_state) if random_state is not None else None,
            "max_iter": int(max_iter),
            "n_init": int(n_init),
            "metric": default_metric,
            "dissimilarity": default_dissimilarity,
        }

        # Allow kwargs to override the minimal params if desired
        if kwargs:
            self.params.update(kwargs)

        self.model = None
        self.projection = None

    def fit_transform(self, dataset: Dataset, **kwargs: Any) -> np.ndarray:
        """
        Perform MDS projection using only a custom DistanceMeasure.
        """
        self.params.update(kwargs)


        X = np.asarray(dataset.get_data(), dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one sample.")

        # Build kwargs for sklearn.MDS from the adapter params
        mds_kwargs = dict(self.params)
        dissimilarity = mds_kwargs.get("dissimilarity", "euclidean")

        # Check if using precomputed distances (custom distance measure)
        if dissimilarity == "precomputed":
            if self.distance_measure is None:
                raise ValueError(
                    "dissimilarity='precomputed' requires a distance_measure parameter. "
                    "Either provide distance_measure in constructor or use a built-in dissimilarity"
                )
            # Use precomputed distances
            dist_matrix = build_distance_matrix(X, self.distance_measure)
            self.model = MDS(**mds_kwargs)
            self.projection = self.model.fit_transform(dist_matrix)
        else:
            # Using sklearn's built-in distance computation
            if self.distance_measure is not None:
                raise ValueError(
                    "distance_measure provided but dissimilarity is not 'precomputed'. "
                    "To use custom distance measures, don't override dissimilarity parameter"
                )
            # No DistanceMeasure: pass raw X to sklearn's MDS
            self.model = MDS(**mds_kwargs)
            self.projection = self.model.fit_transform(X)

        return np.asarray(self.projection)
