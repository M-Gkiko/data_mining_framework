from typing import Any, Optional
import numpy as np
from sklearn.manifold import TSNE
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure
from ...core.dimensionality_reduction import DimensionalityReduction
from ...utils.distance_utils import build_distance_matrix


class TSNEProjection(DimensionalityReduction):
    """
    Adapter for sklearn's t-SNE restricted to custom DistanceMeasure implementations.

    - Uses both our DistanceMeasure from our framework or sklearn's built-in metrics.
    - Returns a 2D numpy array (rows = samples, columns = components).
    """

    def __init__(self, distance_measure: Optional[DistanceMeasure] = None, **kwargs: Any):
        # Store distance measure (may be None)
        self.distance_measure = distance_measure

        # Default t-SNE parameters
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
            "n_components": 2,
            "perplexity": 30,
            "learning_rate": "auto",
            "init": "random",
            "random_state": 42,
            "metric": default_metric,
        }

        # Apply any provided kwargs directly to adapter params (no splitting)
        if kwargs:
            self.params.update(kwargs)

        self.model = None
        self.projection = None

    def fit_transform(self, dataset: Dataset, **kwargs: Any) -> np.ndarray:
        """
        Perform t-SNE projection using a custom DistanceMeasure.
        """
        # Merge any per-call kwargs into the stored params
        self.params.update(kwargs)

        X = np.asarray(dataset.get_data(), dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one sample.")

        # Prepare kwargs to forward to sklearn TSNE (adapter params contain all)
        tsne_kwargs = dict(self.params)

        metric = tsne_kwargs.get("metric", "euclidean")

        # Validate metric vs provided DistanceMeasure
        if metric == "precomputed":
            if self.distance_measure is None:
                raise ValueError(
                    "metric='precomputed' requires a distance_measure parameter. "
                    "Either provide distance_measure in constructor or use a built-in metric"
                )
            # build distance matrix and ensure metric is precomputed
            dist_matrix = build_distance_matrix(X, self.distance_measure)
            tsne_kwargs["metric"] = "precomputed"
            model = TSNE(**tsne_kwargs)
            proj = model.fit_transform(dist_matrix)
            self.projection = np.asarray(proj)
        else:
            # If a custom DistanceMeasure was given but metric is not precomputed, that's invalid
            if self.distance_measure is not None:
                raise ValueError(
                    "distance_measure provided but using built-in metric. "
                    "To use custom distance measures, set metric='precomputed' or don't provide metric parameter"
                )
            # No DistanceMeasure: pass raw X to TSNE
            model = TSNE(**tsne_kwargs)
            proj = model.fit_transform(X)
            self.projection = np.asarray(proj)

        return np.asarray(self.projection)
