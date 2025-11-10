from abc import ABC, abstractmethod
import numpy as np
from ..core.dataset import Dataset


class DRQualityMeasure(ABC):
    """
    Abstract base class for all Dimensionality Reduction (DR) quality measures.

    This meta-component defines a consistent interface for evaluating
    the quality of a dimensionality reduction technique.

    Each subclass must implement the `evaluate()` method, which computes
    a numerical quality score (float) given the original dataset and
    the reduced (projected) data.

    Common examples of DR quality measures include:
    - Reconstruction Error (information loss)
    - Trustworthiness (local structure preservation)
    - Continuity (global neighborhood preservation)
    """

    @abstractmethod
    def evaluate(self, dataset: Dataset, reduced_data: np.ndarray) -> float:
        """
        Evaluate the quality of a dimensionality reduction result.

        Args:
            dataset (Dataset): The original high-dimensional dataset.
            reduced_data (np.ndarray): The reduced (projected) data
                with shape (n_samples, n_components).

        Returns:
            float: A quality score. Higher or lower values depend on
                   the specific metric (e.g., lower is better for
                   reconstruction error, higher is better for trustworthiness).
        """
        pass
