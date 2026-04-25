"""
OmniStack-RS — Stage 5: Lloyd-Max INT4 Codebook Calibration

Finds the 16 optimal centroids for INT4 quantization of Hadamard-rotated
KV activations. The normalized WHT sphericizes the distribution, so the
optimal codebook approximates a Gaussian quantizer with equal-probability
cells — but Lloyd-Max on real activations achieves strictly lower MSE than
any analytic approximation.

Algorithm: Lloyd-Max iterative quantizer design.
  1. Initialize 16 centroids by partitioning the sorted data into 16 equal bins.
  2. Assign each sample to its nearest centroid (Voronoi partition).
  3. Update each centroid to the mean of its assigned samples.
  4. Repeat until max centroid shift < convergence_tol or max_iters reached.

Invariants:
  - Output centroids are sorted ascending (enables binary search at runtime).
  - Two values per byte (nibble): indices ∈ [0, 15], stored as uint8.
  - The codebook lives in 16 × 4 bytes = 64 bytes — fits in one cache line
    and is broadcast to SRAM once per Triton block.
"""

from __future__ import annotations

import torch


N_CENTROIDS: int = 16          # 2^4 for INT4
_DEFAULT_MAX_ITERS: int = 300
_DEFAULT_TOL: float = 1e-6


class LloydMaxCalibrator:
    """
    Fit a sorted 16-centroid codebook from a flat sample of activations.

    Usage:
        cal = LloydMaxCalibrator()
        cal.fit(rotated_kv_activations.reshape(-1))
        codebook = cal.codebook   # (16,) float32, sorted ascending
    """

    def __init__(
        self,
        n_centroids: int = N_CENTROIDS,
        max_iters: int = _DEFAULT_MAX_ITERS,
        convergence_tol: float = _DEFAULT_TOL,
    ) -> None:
        if n_centroids != 16:
            raise ValueError(
                f"n_centroids must be 16 for INT4 nibble packing; got {n_centroids}"
            )
        self.n_centroids = n_centroids
        self.max_iters = max_iters
        self.convergence_tol = convergence_tol
        self._codebook: torch.Tensor | None = None

    @property
    def codebook(self) -> torch.Tensor:
        if self._codebook is None:
            raise RuntimeError("LloydMaxCalibrator.fit() must be called before accessing codebook")
        return self._codebook

    def fit(self, samples: torch.Tensor) -> "LloydMaxCalibrator":
        """
        Run Lloyd-Max iteration on a flat (N,) sample of activations.

        Args:
            samples: 1-D float tensor of activation values (any dtype; promoted to float32).
        Returns:
            self (fluent API)
        """
        x = samples.reshape(-1).float()
        if x.numel() < self.n_centroids:
            raise ValueError(
                f"Need at least {self.n_centroids} samples; got {x.numel()}"
            )

        # Initialize: partition sorted data into n_centroids equal bins, take mean of each.
        x_sorted = x.sort().values
        bin_size = x_sorted.shape[0] // self.n_centroids
        centroids = torch.stack([
            x_sorted[i * bin_size : (i + 1) * bin_size].mean()
            for i in range(self.n_centroids)
        ])  # (16,) — already sorted ascending due to sorted x

        for _ in range(self.max_iters):
            # Assign each sample to its nearest centroid.
            # distances: (N, 16); argmin over dim-1 → cluster indices (N,)
            dists = (x.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # (N, 16)
            assignments = dists.argmin(dim=1)                        # (N,)

            # Update: new centroid = mean of assigned samples.
            new_centroids = centroids.clone()
            for c in range(self.n_centroids):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = x[mask].mean()
                # If centroid is empty, leave it in place — avoids codebook collapse.

            shift = (new_centroids - centroids).abs().max().item()
            centroids = new_centroids
            if shift < self.convergence_tol:
                break

        # Ensure sorted ascending so binary search works at inference.
        self._codebook = centroids.sort().values.contiguous()
        return self

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize x to INT4 indices using the fitted codebook.

        Args:
            x: (...) float tensor
        Returns:
            (indices, dequantized):
                indices     — (...) uint8 tensor, values ∈ [0, 15]
                dequantized — (...) float32 tensor (nearest centroid values)
        """
        cb = self.codebook  # (16,)
        flat = x.reshape(-1).float()
        dists = (flat.unsqueeze(1) - cb.unsqueeze(0)).abs()  # (N, 16)
        idx = dists.argmin(dim=1).to(torch.uint8)            # (N,)
        dequant = cb[idx.long()]                              # (N,)
        return idx.reshape(x.shape), dequant.reshape(x.shape)


def calibrate_codebook(
    samples: torch.Tensor,
    max_iters: int = _DEFAULT_MAX_ITERS,
    convergence_tol: float = _DEFAULT_TOL,
) -> torch.Tensor:
    """
    Convenience wrapper: fit and return the (16,) sorted codebook.

    Args:
        samples: any-shape float tensor of activation samples
    Returns:
        (16,) float32 sorted codebook
    """
    return LloydMaxCalibrator(
        max_iters=max_iters,
        convergence_tol=convergence_tol,
    ).fit(samples).codebook


def calibrate_per_group(
    samples_by_group: list[torch.Tensor],
    max_iters: int = _DEFAULT_MAX_ITERS,
    convergence_tol: float = _DEFAULT_TOL,
) -> torch.Tensor:
    """
    Fit one Lloyd-Max codebook per KV head group.

    Each group sees a different activation distribution (e.g. "Action Ad" heads
    concentrate near different centroids than "Romance Ad" heads), so per-group
    codebooks reduce quantization MSE compared to a single shared codebook.

    Args:
        samples_by_group: list of n_groups tensors, each (N_g,) or (N_g, D)
                          flat sample of activations for that group.
    Returns:
        (n_groups, 16) float32 — row g is the sorted codebook for group g.
    """
    if len(samples_by_group) == 0:
        raise ValueError("samples_by_group must be non-empty")
    codebooks = [
        calibrate_codebook(s, max_iters=max_iters, convergence_tol=convergence_tol)
        for s in samples_by_group
    ]
    return torch.stack(codebooks)  # (n_groups, 16)
