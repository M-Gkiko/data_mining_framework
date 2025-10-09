from typing import List
import numpy as np
from sklearn.metrics import silhouette_score
from core.clustering_quality_measure import ClusteringQualityMeasure
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure
from utils.distance_utils import build_distance_matrix


class Silhouette(ClusteringQualityMeasure):
    def __init__(self, distance_measure: DistanceMeasure):
        """
        Initialize with a given distance measure.
        """
        self.distance_measure = distance_measure

    def evaluate(self, dataset: Dataset, labels: List[int]) -> float:
        """
        Evaluate clustering quality using the Silhouette score.
        """
        data = dataset.get_data()
        if data is None or len(data) == 0:
            raise ValueError("Dataset is empty or invalid.")

        if len(labels) != dataset.get_rows():
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match dataset size ({dataset.get_rows()})."
            )

        data = np.asarray(data)
        labels = np.asarray(labels)

        # Compute using Euclidean or a custom distance
        if self.distance_measure.get_name().lower() != "euclidean":
            distance_matrix = build_distance_matrix(data, self.distance_measure)
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
        else:
            score = silhouette_score(data, labels, metric="euclidean")

        return float(score)