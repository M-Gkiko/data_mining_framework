from typing import List
import numpy as np
from sklearn.metrics import calinski_harabasz_score

from core.clustering_quality_measure import ClusteringQualityMeasure
from core.dataset import Dataset


class CalinskiHarabaszIndex(ClusteringQualityMeasure):
    """
    
    The Calinski-Harabasz Index is also known as the Variance Ratio Criterion.
    - Higher values indicate better clustering (better defined clusters).
    - It measures the ratio of between-cluster dispersion to within-cluster dispersion.
    - Formula: CHI = (B_k / W_k) * ((N - k) / (k - 1))
      where B_k = between-cluster sum of squares, W_k = within-cluster sum of squares,
      N = number of points, k = number of clusters.
    
    """

    def evaluate(self, dataset: Dataset, labels: List[int]) -> float:
        """
        Evaluate clustering quality using Calinski-Harabasz Index.
        
        Args:
            dataset (Dataset): The original dataset that was clustered
            labels (List[int]): Cluster labels assigned to each data point
            
        Returns:
            float: Calinski-Harabasz Index score (higher values = better clustering)
            
        Raises:
            ValueError: If dataset is empty, labels length doesn't match dataset size,
                       or if there's insufficient cluster diversity
        """

        data = dataset.get_data()
        if data is None or len(data) == 0:
            raise ValueError("Dataset is empty or invalid.")
        
        if len(labels) != dataset.get_rows():
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match dataset size ({dataset.get_rows()})."
            )
        
        X = np.asarray(data)
        labels_array = np.asarray(labels)
        
        unique_labels = np.unique(labels_array)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            raise ValueError(
                f"Calinski-Harabasz Index requires at least 2 clusters, got {n_clusters}."
            )
        
        try:
            score = calinski_harabasz_score(X, labels_array)
            return float(score)
        except Exception as e:
            raise ValueError(f"Failed to compute Calinski-Harabasz Index: {str(e)}")
