import numpy as np
from sklearn.decomposition import PCA
from ...core.dimensionality_reduction import DimensionalityReduction
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure
from ...utils.distance_utils import build_distance_matrix

class SammonMapping(DimensionalityReduction):
    """
    Stable Sammon Mapping for nonlinear dimensionality reduction.
    """
    def __init__(self, distance_measure: DistanceMeasure = None, n_components: int = 2,
                 max_iter: int = 300, tol: float = 1e-9, learning_rate: float = 0.01,
                 random_state: int = None, epsilon: float = 1e-8, **kwargs):
        self.distance_measure = distance_measure
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.epsilon = epsilon

    def fit_transform(self, dataset: Dataset, **kwargs) -> np.ndarray:
        # Extract data matrix
        if hasattr(dataset, 'get_data') and callable(dataset.get_data):
            X = dataset.get_data()
        elif isinstance(dataset, np.ndarray):
            X = dataset
        else:
            X = np.asarray(dataset)

        if X.ndim != 2:
            raise ValueError(f"Input data must be 2D array, got shape {X.shape}")

        n_samples = X.shape[0]
        eps = self.epsilon

        # Compute distance matrix in original space
        if self.distance_measure is not None:
            D = build_distance_matrix(X, self.distance_measure)
        else:
            D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

        # Avoid zero distances
        D = D + np.eye(n_samples) * eps
        D[D < eps] = eps

        D_sum = D.sum() / 2

        # Initialize Y with PCA for stability
        Y = PCA(n_components=self.n_components).fit_transform(X)

        for it in range(self.max_iter):
            # Compute distances in projected space
            d = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
            d = d + np.eye(n_samples) * eps
            d[d < eps] = eps

            # Compute Sammon stress
            delta = D - d
            ratio = delta / (D + eps)
            ratio = np.clip(ratio, -1e5, 1e5)  # prevent extreme gradients
            E = (ratio ** 2).sum() / D_sum

            if np.isnan(E) or np.isinf(E):
                print(f"Iteration {it}: E exploded, stopping early.")
                break

            # Compute gradient
            grad = np.zeros_like(Y)
            for i in range(n_samples):
                diff = Y[i] - Y
                d_i = np.maximum(d[i], eps)
                D_i = np.maximum(D[i], eps)
                ratio_i = (D_i - d_i) / (D_i * d_i)
                ratio_i = np.clip(ratio_i, -1e5, 1e5)
                grad[i] = (ratio_i[:, None] * diff).sum(axis=0)
            grad *= -2 / D_sum

            # Update Y
            Y_new = Y - self.learning_rate * grad

            # Stop if NaNs are produced
            if np.isnan(Y_new).any() or np.isinf(Y_new).any():
                print(f"Iteration {it}: Y exploded, stopping early.")
                break

            # Check convergence
            if np.linalg.norm(Y_new - Y) < self.tol:
                break

            Y = Y_new

        return Y
