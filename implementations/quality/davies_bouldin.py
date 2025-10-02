# Davies-Bouldin quality measure implementation
from typing import List
import numpy as np
from sklearn.metrics import davies_bouldin_score

from core.quality_measure import QualityMeasure
from core.dataset import Dataset


class DaviesBouldinIndex(QualityMeasure):
    """
    Implementation of the Davies–Bouldin Index (DBI) quality measure.

    - Lower values indicate better clustering (0 = perfect).
    - DBI is defined as the average similarity between clusters,
      where similarity is a function of within-cluster scatter
      and between-cluster separation.
    """

    def evaluate(self, dataset: Dataset, labels: List[int]) -> float:
        data = dataset.get_data()
        if data is None or len(data) == 0:
            raise ValueError("Dataset is empty or invalid.")

        if len(labels) != dataset.get_rows():
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match dataset size ({dataset.get_rows()})."
            )

        # Convert to numpy array
        X = np.asarray(data)

        # Compute Davies–Bouldin Index using sklearn
        score = davies_bouldin_score(X, labels)
        return float(score)
