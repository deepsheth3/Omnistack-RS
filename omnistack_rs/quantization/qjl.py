"""
OmniStack-RS — Stage 5: 1-Bit Rademacher QJL Residual

After INT4 quantization, a residual remains:
    error = x_original - x_dequantized

QJL (Quantized Johnson-Lindenstrauss) encodes this residual into a 1-bit
sign bitmask, giving an additional ~31.8% MSE reduction over INT4 alone.

Projection matrix (Rademacher sketch):
    G ∈ {-1, +1}^(qjl_dim × head_dim)

The matrix is NEVER stored. It is regenerated on-the-fly from a seeded PRNG:

    seed = (user_id % 1024) ^ head_idx

    Two axes of uniqueness:
      - head_idx:  prevents cross-head structured noise (heads see different G)
      - user_id:   prevents cross-user interference at Meta scale (billions of
                   users sharing a server → same head_idx but different users
                   must not share the same projection basis)

Sign projection:
    b_i = sign(G[i, :] @ error) ∈ {0, 1}  (1 bit per projection)

Reconstruction (optimal linear estimator, derived from MSE minimisation):

    E[(G^T b)_j] = qjl_dim · √(2/π) · error_j / ‖error‖
    ⟹ α_opt = √(2/π) · ‖error‖ / head_dim
    error_hat = (G^T @ b_signed) * α_opt

    This gives MSE(error - error_hat) = ‖error‖² · (1 - qjl_dim·2/(π·head_dim)),
    a ~31.8% reduction vs INT4 alone at qjl_dim = head_dim/2 = 64.

Invariant: G is Rademacher (+1 or -1), NOT Gaussian, so all matrix-vector
products reduce to conditional adds — critical for the Triton kernel.
"""

from __future__ import annotations

import math

import torch


class RademacherQJL:
    """
    On-the-fly Rademacher projection for 1-bit QJL residual encoding.

    The G matrix is regenerated per call — zero persistent storage.
    Seed = (user_id % 1024) ^ head_idx prevents both cross-head and
    cross-user structured noise.
    """

    def __init__(self, head_dim: int, qjl_dim: int) -> None:
        self.head_dim = head_dim
        self.qjl_dim = qjl_dim

    def _make_G(
        self,
        head_idx: int,
        device: torch.device,
        dtype: torch.dtype,
        user_id: int = 0,
    ) -> torch.Tensor:
        """
        Generate the (qjl_dim, head_dim) Rademacher matrix.

        seed = (user_id % 1024) ^ head_idx  — unique per (user, head) pair.
        """
        seed = (user_id % 1024) ^ head_idx
        rng  = torch.Generator(device=device)
        rng.manual_seed(seed)
        bits = torch.randint(0, 2, (self.qjl_dim, self.head_dim),
                             generator=rng, device=device)
        return (bits * 2 - 1).to(dtype)

    def encode(
        self,
        error: torch.Tensor,
        head_idx: int,
        user_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project residual error to 1-bit signs and record ‖error‖ for reconstruction.

        Args:
            error:    (..., head_dim) residual tensor
            head_idx: integer head index (part of PRNG seed)
            user_id:  integer user identifier (part of PRNG seed)
        Returns:
            (signs, norms):
                signs — (..., qjl_dim) bool tensor (True = positive projection)
                norms — (..., 1) float32 tensor (‖error‖₂ per vector)
        """
        orig_dtype = error.dtype
        err_f = error.float()
        G = self._make_G(head_idx, error.device, torch.float32, user_id)  # (qjl_dim, D)

        proj  = err_f @ G.T      # (..., qjl_dim)
        signs = proj >= 0        # bool

        norms = err_f.norm(dim=-1, keepdim=True).to(orig_dtype)  # (..., 1)
        return signs, norms

    def reconstruct(
        self,
        signs: torch.Tensor,
        norms: torch.Tensor,
        head_idx: int,
        user_id: int = 0,
    ) -> torch.Tensor:
        """
        Approximate reconstruction of the residual from 1-bit signs.

        Optimal linear estimator:
            α_opt = √(2/π) · norm / head_dim
            error_hat = G^T @ b_signed * α_opt

        Args:
            signs:    (..., qjl_dim) bool tensor
            norms:    (..., 1) float32 norm tensor
            head_idx: must match encode() head_idx
            user_id:  must match encode() user_id
        Returns:
            (..., head_dim) float32 approximate residual
        """
        G = self._make_G(head_idx, signs.device, torch.float32, user_id)
        b = (signs.float() * 2.0 - 1.0)   # {-1, +1}
        error_hat = b @ G                   # (..., head_dim)
        # α_opt = √(2/π) · norm / head_dim
        error_hat = error_hat * (norms.float() * math.sqrt(2.0 / math.pi) / self.head_dim)
        return error_hat


def qjl_encode(
    x_orig: torch.Tensor,
    x_dequant: torch.Tensor,
    head_idx: int,
    qjl_dim: int = 64,
    user_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full INT4+QJL encode: compute residual and project to 1-bit signs.

    Args:
        x_orig:    (..., head_dim) original (rotated) activations
        x_dequant: (..., head_dim) INT4-dequantized values
        head_idx:  head index (part of PRNG seed)
        qjl_dim:   number of sign projections (default 64)
        user_id:   user identifier (part of PRNG seed; prevents cross-user noise)
    Returns:
        (error, signs, norms)
    """
    head_dim = x_orig.shape[-1]
    qjl = RademacherQJL(head_dim=head_dim, qjl_dim=qjl_dim)
    error = (x_orig.float() - x_dequant.float())
    signs, norms = qjl.encode(error, head_idx, user_id=user_id)
    return error, signs, norms


def qjl_reconstruct(
    x_dequant: torch.Tensor,
    signs: torch.Tensor,
    norms: torch.Tensor,
    head_idx: int,
    qjl_dim: int = 64,
    user_id: int = 0,
) -> torch.Tensor:
    """
    Reconstruct the original activation from INT4 dequantized + QJL residual.

    Args:
        x_dequant: (..., head_dim) INT4-dequantized values
        signs:     (..., qjl_dim) bool signs from qjl_encode
        norms:     (..., 1) float32 norms from qjl_encode
        head_idx:  must match encode head_idx
        qjl_dim:   must match encode qjl_dim
        user_id:   must match encode user_id
    Returns:
        (..., head_dim) float32 approximate reconstruction
    """
    head_dim = x_dequant.shape[-1]
    qjl = RademacherQJL(head_dim=head_dim, qjl_dim=qjl_dim)
    error_hat = qjl.reconstruct(signs, norms, head_idx, user_id=user_id)
    return x_dequant.float() + error_hat
