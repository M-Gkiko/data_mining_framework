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
        # Choose default metric depending on whether a DistanceMeasure was provided
        if distance_measure is not None:
            default_metric = "precomputed"
        else:
            default_metric = kwargs.pop("metric", "euclidean")

        if "metric" in kwargs and distance_measure is not None:
            raise ValueError(
                "Cannot override metric when using custom distance_measure. "
                "Custom distance measures always use metric='precomputed'"
            )

        self.params = {
            "n_components": int(n_components),
            "random_state": int(random_state) if random_state is not None else None,
            "max_iter": int(max_iter),
            "n_init": int(n_init),
            "metric": default_metric,
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
        metric = mds_kwargs.get("metric", "euclidean")

        # Validate metric vs provided DistanceMeasure
        if metric == "precomputed":
            if self.distance_measure is None:
                raise ValueError(
                    "metric='precomputed' requires a distance_measure parameter. "
                    "Either provide distance_measure in constructor or use a built-in metric"
                )
            # use precomputed distances
            dist_matrix = build_distance_matrix(X, self.distance_measure)
            mds_kwargs["dissimilarity"] = "precomputed"
            self.model = MDS(**mds_kwargs)
            self.projection = self.model.fit_transform(dist_matrix)
        else:
            # If a custom DistanceMeasure was given but metric is not precomputed, that's invalid
            if self.distance_measure is not None:
                raise ValueError(
                    "distance_measure provided but using built-in metric. "
                    "To use custom distance measures, set metric='precomputed' or don't provide metric parameter"
                )
            # No DistanceMeasure: pass raw X to sklearn's MDS
            self.model = MDS(**mds_kwargs)
            self.projection = self.model.fit_transform(X)

        return np.asarray(self.projection)
