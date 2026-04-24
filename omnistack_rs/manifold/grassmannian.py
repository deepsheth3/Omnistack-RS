"""
OmniStack-RS — Stage 4: Grassmannian Manifold Projector

Implements the Gr(k, D) Grassmann manifold projection that compresses
128-dim user embeddings to a low-rank subspace capturing entertainment
taste. This is the first half of the 12.8× VRAM reduction:
  128 → 32 active dims = 4× compression factor.

Mathematical basis:
  The Grassmannian Gr(k, D) is the space of k-dimensional linear subspaces
  of R^D. We find the optimal k-subspace via truncated SVD (equivalent to
  PCA), which minimizes reconstruction error under the Frobenius norm.
  The basis U ∈ R^{D×k} satisfies U^T U = I_k (columns are orthonormal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SVDProjectionResult:
    """Diagnostics returned by GrassmannianProjector.fit()."""
    U: np.ndarray                       # (D, k) orthonormal basis vectors
    singular_values: np.ndarray         # (k,) descending singular values
    explained_variance_ratio: np.ndarray   # (k,) per-component fraction of total variance
    cumulative_variance_ratio: np.ndarray  # (k,) cumulative sum of explained_variance_ratio
    total_components_available: int     # min(N, D) — max possible rank

    def print_summary(self, label: str = "") -> None:
        """Pretty-print a variance breakdown to stdout."""
        header = f"── Explained Variance Ratio {label}".rstrip()
        print(header + "─" * max(0, 60 - len(header)))
        for i, (evr, cumevr) in enumerate(
            zip(self.explained_variance_ratio, self.cumulative_variance_ratio)
        ):
            bar = "█" * int(evr * 50)
            print(f"  Component {i+1:2d}: {evr:.4f}  (Σ={cumevr:.4f})  {bar}")
        k = len(self.explained_variance_ratio)
        total_var = self.cumulative_variance_ratio[-1]
        print(f"\n  ✓ {k} components capture {total_var:.1%} of total variance")
        print(f"  ✓ Dimensionality: {self.U.shape[0]}D → {k}D")
        print(f"  ✓ Compression factor from manifold alone: {self.U.shape[0] / k:.1f}×")


class GrassmannianProjector:
    """
    Projects user embeddings onto the Gr(k, D) Grassmann manifold.

    Usage (Stage 4 of the 6-Stage Firewall):
        projector = GrassmannianProjector(ambient_dim=128, rank=32)
        result = projector.fit(calibration_embeddings)   # one-time calibration
        result.print_summary()

        z = projector.project(user_embeddings)    # (N, 128) → (N, 32)
        x_approx = projector.lift(z)              # (N, 32)  → (N, 128)

    The basis U is fit via truncated SVD, which is optimal (Eckart-Young theorem):
    it minimizes the Frobenius reconstruction error among all rank-k projections.
    """

    def __init__(self, ambient_dim: int, rank: int) -> None:
        if rank > ambient_dim:
            raise ValueError(f"rank {rank} must be ≤ ambient_dim {ambient_dim}")
        self.D = ambient_dim
        self.k = rank
        self.U: Optional[np.ndarray] = None      # (D, k) fit basis
        self._mean: Optional[np.ndarray] = None  # (D,) training mean for centering
        self._fit_result: Optional[SVDProjectionResult] = None

    @property
    def is_fitted(self) -> bool:
        return self.U is not None

    def fit(self, embeddings: np.ndarray) -> SVDProjectionResult:
        """
        Fit the Grassmannian basis to a calibration set of user embeddings.

        Args:
            embeddings: (N, D) float32 array, N ≥ k.

        Returns:
            SVDProjectionResult with per-component explained variance ratios.
            Use result.cumulative_variance_ratio[-1] to verify dimensionality
            collapse: how much information is preserved at rank k.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        N, D = embeddings.shape
        if D != self.D:
            raise ValueError(f"Expected embeddings of dim {self.D}, got {D}")
        if N < self.k:
            raise ValueError(f"Need ≥ {self.k} samples to fit rank-{self.k} basis, got {N}")

        # Center: remove mean so SVD captures directional variance, not offset
        # Use float64 throughout SVD for numerical stability with large matrices
        self._mean = embeddings.mean(axis=0).astype(np.float64)
        centered = embeddings.astype(np.float64) - self._mean

        # Economy SVD: U_svd (N,r), s (r,), Vt (r,D) where r = min(N,D)
        # Vt rows are right singular vectors = principal directions in R^D
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)

        # Top-k right singular vectors form our Grassmannian basis
        self.U = Vt[: self.k].T.astype(np.float64)  # (D, k) — columns are orthonormal

        # Explained variance: σ_i² / Σ σ_j²
        variance = s ** 2
        total_var = variance.sum()
        evr = variance[: self.k] / total_var
        cumevr = np.cumsum(evr)

        self._fit_result = SVDProjectionResult(
            U=self.U.astype(np.float32),
            singular_values=s[: self.k].astype(np.float32),
            explained_variance_ratio=evr.astype(np.float32),
            cumulative_variance_ratio=cumevr.astype(np.float32),
            total_components_available=len(s),
        )
        return self._fit_result

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project (N, D) embeddings onto the k-dim Grassmannian subspace.

        Returns (N, k) float32. This is the compressed user representation
        that feeds into Stage 5 KV cache compression.
        """
        self._require_fit()
        x = np.asarray(x, dtype=np.float64)
        # errstate: Apple Accelerate BLAS can raise spurious IEEE 754 signals
        # during DGEMM on denormal inputs even when the result is finite.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            z = (x - self._mean) @ self.U
        return z.astype(np.float32)  # (N, k)

    def lift(self, z: np.ndarray) -> np.ndarray:
        """
        Lift (N, k) Grassmannian coordinates back to (N, D) ambient space.

        The lifted embedding is the best rank-k approximation of the original.
        Reconstruction error measures information loss from pruning.
        """
        self._require_fit()
        z = np.asarray(z, dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            x_approx = z @ self.U.T + self._mean
        return x_approx.astype(np.float32)  # (N, D)

    def reconstruction_error(self, x: np.ndarray) -> float:
        """
        Relative Frobenius reconstruction error: ‖x − lift(project(x))‖_F / ‖x‖_F.

        Quantifies information lost by the manifold compression.
        Target for production: < 0.10 (< 10% relative error at rank=32).
        """
        x = np.asarray(x, dtype=np.float32)
        x_recon = self.lift(self.project(x))
        error = np.linalg.norm(x - x_recon, "fro")
        total = np.linalg.norm(x, "fro")
        return float(error / (total + 1e-8))

    def find_rank_for_variance(
        self,
        embeddings: np.ndarray,
        target_variance: float = 0.95,
    ) -> int:
        """
        Find the minimum rank needed to explain `target_variance` of total variance.

        Does not modify the projector state (no side effects on self.U).
        Useful for choosing manifold_rank without trial-and-error.

        Example:
            rank_95 = projector.find_rank_for_variance(X, target_variance=0.95)
            print(f"Need rank {rank_95} to capture 95% of variance")
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        centered = embeddings - embeddings.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        variance = s ** 2
        cumulative = np.cumsum(variance) / (variance.sum() + 1e-12)
        rank = int(np.searchsorted(cumulative, target_variance)) + 1
        return min(rank, self.D)

    def _require_fit(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("GrassmannianProjector must be fit() before project() or lift().")
