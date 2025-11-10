from typing import Any, Optional
import warnings
import numpy as np

from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure
from ...core.dimensionality_reduction import DimensionalityReduction
from ...utils.distance_utils import build_distance_matrix


class SammonMapping(DimensionalityReduction):
    """Sammon mapping (adapted/ported)

        Ported/adapted from:
            Tom J. Pollard (tom.pollard.11@ucl.ac.uk) — Python port of the MATLAB
            implementation by Gavin C. Cawley and Nicola L. C. Talbot.

        Original algorithm:
            J. W. Sammon Jr., "A Nonlinear Mapping for Data Structure Analysis",
            IEEE Transactions on Computers, 1969.

        This file includes code adapted from the Sammon Python port (copyright
        Gavin C. Cawley, 2007) and is used here with attribution. See project
        """

    def __init__(
        self,
        distance_measure: Optional[DistanceMeasure] = None,
        jitter_scale: Optional[float] = None,
        step_scale: float = 1.0,
        max_step_norm: float = 1.0,
        **kwargs: Any,
    ) -> None:
        # store distance measure and explicit control parameters
        self.distance_measure = distance_measure
        # user-controllable jitter (None = automatic small jitter based on data scale)
        self.jitter_scale = None if jitter_scale is None else float(jitter_scale)

        # Defaults (mirror the original implementation defaults)
        self.params = {
            "n_components": 2,
            "maxiter": 500,
            "tolfun": 1e-9,
            "maxhalves": 20,
            "init": "pca",
            "display": 0,
            # step scaling factor applied to computed step (1.0 = no extra scaling)
            "step_scale": float(step_scale),
            # maximum allowed L2 norm for the flattened step vector; set to a
            # finite value to avoid unbounded updates (can be tuned per dataset)
            "max_step_norm": float(max_step_norm),
        }
        # Merge any other user-supplied kwargs
        self.params.update(kwargs)

    def fit_transform(self, dataset: Dataset, **kwargs: Any) -> np.ndarray:
        """Compute a Sammon mapping for `dataset` using the provided DistanceMeasure.

        Returns only the projection array (n_samples x n_components) to match
        the project's DimensionalityReduction interface.
        """
        # Update parameters
        self.params.update(kwargs)

        if self.distance_measure is None:
            raise ValueError("SammonMapping requires a DistanceMeasure instance in the constructor.")

        X = np.asarray(dataset.get_data(), dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one sample.")

        n_components = int(self.params["n_components"])
        maxiter = int(self.params["maxiter"])
        tolfun = float(self.params["tolfun"])
        maxhalves = int(self.params["maxhalves"])
        init = str(self.params.get("init", "pca"))
        display = int(self.params.get("display", 0))

        # Build pairwise distance matrix using project's utility
        D = build_distance_matrix(X, self.distance_measure)

        N = D.shape[0]
        if N != X.shape[0]:
            raise ValueError("Distance matrix size does not match number of samples")

        if np.count_nonzero(np.diagonal(D)) > 0:
            raise ValueError("The diagonal of the dissimilarity matrix must be zero")

        # If there are zero or negative off-diagonal distances (duplicate
        # or extremely-close points), add tiny jitter to the input data
        # and rebuild the distance matrix. Jitter is preferable to clamping
        # distances because it preserves the distance metric semantics.
        offdiag_mask = ~np.eye(N, dtype=bool)
        nonpos = int(np.sum((D <= 0) & offdiag_mask))
        if nonpos > 0:
            msg = (
                f"SammonMapping: {nonpos} non-positive off-diagonal distances found; "
                "adding small jitter to input data to break ties and rebuilding distance matrix"
            )
            # single warning emitted via warnings.warn
            warnings.warn(msg, UserWarning)

            # Compute jitter scale proportional to data scale or use user-specified value
            data_scale = float(np.std(X)) if np.std(X) != 0 else 1.0
            if self.jitter_scale is None:
                jitter_scale = 1e-8 * data_scale
            else:
                jitter_scale = float(self.jitter_scale)

            rng = np.random.default_rng()
            X = X + rng.normal(loc=0.0, scale=jitter_scale, size=X.shape)

            # Rebuild distance matrix using jittered data
            D = build_distance_matrix(X, self.distance_measure)

            # Recompute helper values
            N = D.shape[0]
            if np.count_nonzero(np.diagonal(D)) > 0:
                raise ValueError("The diagonal of the dissimilarity matrix must be zero after jittering")

        scale = 0.5 / D.sum()
        # Add identity to avoid division by zero later (matches original)
        D = D + np.eye(N)

        Dinv = 1.0 / D

        # Initialize embedding
        if init == "pca":
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            # scale columns of U by singular values and take first n_components
            y = U[:, :n_components] * S[:n_components]
        elif init == "random":
            rng = np.random.default_rng()
            y = rng.normal(0.0, 1.0, size=(N, n_components))
        else:
            raise ValueError(f"Unsupported init '{init}'. Use 'pca' or 'random'.")

        # Helper to compute pairwise distances without scipy
        def pairwise_distances(A: np.ndarray) -> np.ndarray:
            dif = A[:, None, :] - A[None, :, :]
            d = np.sqrt(np.maximum(0.0, np.sum(dif * dif, axis=-1)))
            return d

        one = np.ones((N, n_components), dtype=float)
        d = pairwise_distances(y) + np.eye(N)
        # ensure distances are strictly positive to avoid divide-by-zero
        dist_eps = max(1e-8, np.finfo(float).eps)
        if np.any(d <= 0):
            d = np.where(d <= 0, dist_eps, d)
        dinv = 1.0 / d
        delta = D - d
        E = np.sum((delta ** 2) * Dinv)
        # optimizer controls from params
        step_scale = float(self.params.get("step_scale", 1.0))
        max_step_norm = float(self.params.get("max_step_norm", 1.0))

        # Main optimisation loop
        maxhalves_exceeded_warned = False
        for i in range(maxiter):
            delta = dinv - Dinv
            deltaone = np.dot(delta, one)
            g = np.dot(delta, y) - (y * deltaone)
            dinv3 = dinv ** 3
            y2 = y ** 2
            H = np.dot(dinv3, y2) - deltaone - (2 * y) * np.dot(dinv3, y) + y2 * np.dot(dinv3, one)
            # safe step: prevent division by zero in Hessian diagonal
            H_flat = H.flatten(order='F')
            denom = np.abs(H_flat)
            denom_safe = np.where(denom == 0.0, np.finfo(float).eps, denom)
            s = -g.flatten(order='F') / denom_safe
            # apply optional global step scaling
            if step_scale != 1.0:
                s = s * step_scale

            # sanitize step values
            s = np.where(np.isfinite(s), s, 0.0)

            # clamp overall step magnitude to avoid very large jumps
            s_norm = np.linalg.norm(s)
            if s_norm > 0 and max_step_norm > 0 and s_norm > max_step_norm:
                s = s * (max_step_norm / s_norm)
            y_old = y.copy()

            # Step-halving
            for j in range(maxhalves):
                s_reshape = np.reshape(s, (-1, n_components), order='F')
                y = y_old + s_reshape
                d = pairwise_distances(y) + np.eye(N)
                dinv = 1.0 / d
                delta = D - d
                E_new = np.sum((delta ** 2) * Dinv)
                if E_new < E:
                    break
                s = 0.5 * s

            if j == maxhalves - 1 and not maxhalves_exceeded_warned:
                # Emit a single warning per fit call to avoid spamming notebook output
                warnings.warn('SammonMapping: maxhalves exceeded during step-halving; mapping may not converge', UserWarning)
                maxhalves_exceeded_warned = True

            # TolFun check: guard against non-finite or zero E values
            if not np.isfinite(E) or E == 0.0:
                # If E is not finite, do not attempt relative check; continue
                tol_reached = False
            else:
                tol_reached = abs((E - E_new) / E) < tolfun

            if tol_reached:
                if display:
                    print('TolFun exceeded: Optimisation terminated')
                break

            E = E_new
            if display > 1:
                print('epoch = %d : E = %12.10f' % (i + 1, E * scale))

        if i == maxiter - 1 and display:
            print('Warning: maxiter exceeded. Sammon mapping may not have converged...')

        # Final stress (scaled as in original implementation)
        E = E * scale

        # Only return the embedding to match DimensionalityReduction interface
        return np.asarray(y)