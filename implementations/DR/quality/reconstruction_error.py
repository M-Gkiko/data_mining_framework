import numpy as np
from sklearn.metrics import mean_squared_error
from core.dataset import Dataset
from core.dr_quality_measure import DRQualityMeasure


class ReconstructionError(DRQualityMeasure):
    """
    Measures the reconstruction error between the original data and the data
    reconstructed from its reduced representation (useful for PCA-like techniques).

    Formula:
        RE = mean(||X_original - X_reconstructed||²)
    """

    def evaluate(self, dataset: Dataset, reduced_data: np.ndarray) -> float:
        X = np.asarray(dataset.get_data(), dtype=float)

        # If the DR model provided is invertible, reduced_data can be transformed back
        # For generic DR techniques without inverse_transform, we approximate with PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=reduced_data.shape[1])
        pca.fit(X)
        X_reconstructed = pca.inverse_transform(reduced_data)

        # Compute mean squared reconstruction error
        error = mean_squared_error(X, X_reconstructed)
        return float(error)
