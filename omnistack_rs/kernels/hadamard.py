"""
OmniStack-RS — Stage 5: Walsh-Hadamard Transform (WHT) Rotation

Sphericizes KV activations before INT4 quantization: the uniform-like
output distribution of the WHT makes Lloyd-Max centroids near-optimal,
explaining why WHT + INT4 outperforms plain INT4.

MANIFEST Invariant #1 (NEVER VIOLATE):
    WHT is applied OUTSIDE the attention loop.
    - rotate_queries(q)       — called once per decode step (or at prefill)
    - rotate_kv_cache(k, v)   — called once at prefill; K/V stored permanently
                                 in rotated space for the lifetime of the cache
    The fused Triton kernel (_fused_attention_kernel) NEVER calls any WHT function.

Orthogonality invariant:
    H_D @ H_D^T = D * I_D                          (unnormalized WHT)
    (H_D / √D) @ (H_D / √D)^T = I_D               (normalized WHT = isometry)

    Consequence: dot(Hq/√D, Hk/√D) = dot(q, k)   for all q, k ∈ ℝ^D
    → The attention temperature scale 1/√head_dim is UNCHANGED after rotation.

7-stage Cooley-Tukey butterfly (HEAD_DIM = 128 = 2^7):
    Stage s, half = 1 << s ∈ {1, 2, 4, 8, 16, 32, 64}:
        Partner of element i: j = i XOR half   (flips the s-th bit)
        Upper (bit s = 0): new[i] = x[i] + x[j]
        Lower (bit s = 1): new[i] = x[j] − x[i]
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except (ImportError, Exception):
    _HAS_TRITON = False

# ── Constants ──────────────────────────────────────────────────────────────

HEAD_DIM: int = 128                         # Must equal OmniConfig.head_dim
LOG2_HEAD_DIM: int = 7                      # log₂(128) — number of butterfly stages
_SCALE: float = 1.0 / math.sqrt(HEAD_DIM)  # 1/√128 — isometric normalization


# ── Pure-PyTorch reference (fallback + ground truth) ─────────────────────

def _wht_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Normalized WHT on the last dimension — pure PyTorch.

    Two roles:
      1. Fallback when Triton is unavailable (CPU-only environments).
      2. Ground truth for validating the Triton kernel output.

    Algorithm: iterative Cooley-Tukey.
      At each stage h, view the vector as (D/2h, 2, h):
        • dim-1 index 0 = upper half of each butterfly pair
        • dim-1 index 1 = lower half of each butterfly pair
      Apply: upper' = upper + lower, lower' = upper − lower.
      h doubles each iteration (1 → 2 → 4 → ... → D/2).

    Args:
        x: (..., HEAD_DIM) float tensor of any dtype
    Returns:
        (..., HEAD_DIM) normalized WHT, same dtype as input
    """
    orig_dtype = x.dtype
    x = x.float().clone()
    D = x.shape[-1]
    assert D == HEAD_DIM, f"_wht_torch: expected last dim {HEAD_DIM}, got {D}"
    *batch, _ = x.shape

    h = 1
    while h < D:
        # Reshape to (..., D//(2h), 2, h) — pairs upper/lower within each block
        x = x.reshape(*batch, D // (2 * h), 2, h)
        upper, lower = x[..., 0, :], x[..., 1, :]
        x = torch.stack([upper + lower, upper - lower], dim=-2).reshape(*batch, D)
        h *= 2

    return (x * _SCALE).to(orig_dtype)


# ── Triton kernel ─────────────────────────────────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _wht_kernel(
        X_ptr,
        stride_row: int,         # elements between consecutive rows (= HEAD_DIM for contiguous)
        N_ROWS: int,             # total rows to process
        HEAD_DIM: tl.constexpr, # = 128; drives tl.arange shape (compile-time)
        LOG2_HD:  tl.constexpr, # = 7;   drives tl.static_range bound
        SCALE:    tl.constexpr, # = 1/√128; folded into mul at compile time
    ):
        """
        In-place normalized WHT on one row of a (N_ROWS, HEAD_DIM) tensor.

        Each Triton program handles exactly one row — one attention head vector.

        Butterfly implementation (7 stages, compile-time unrolled):
            For stage s with half = 1 << s:
              1. Store the current 128 values back to DRAM.
              2. Gather partner values: x_partner[i] = x[i XOR half].
              3. Butterfly: upper (bit s=0) → x+x_partner;
                            lower (bit s=1) → x_partner−x.
            The store+gather round-trip mediates the cross-element shuffle
            inside Triton's SPMD model. Each round-trip is 512 bytes; the
            7 stages cost 3.5 KB — negligible vs the attention QK^T compute.

        tl.static_range signals the compiler to unroll this loop at JIT time,
        enabling the backend to schedule loads and arithmetic for max throughput.
        """
        row = tl.program_id(0)
        if row >= N_ROWS:
            return

        base = X_ptr + row * stride_row
        idx = tl.arange(0, HEAD_DIM)  # [0, 1, 2, ..., 127]

        x = tl.load(base + idx)

        # ── 7 butterfly stages — unrolled by compiler ─────────────────────
        for stage in tl.static_range(LOG2_HD):
            half = 1 << stage  # constexpr: 1, 2, 4, 8, 16, 32, 64

            # Write current values so the gather below reads this iteration's state.
            # All 128 stores issue as a single vectorized instruction within this
            # program; the subsequent gather is sequentially ordered after it.
            tl.store(base + idx, x)

            # Partner index: XOR flips the stage-th bit.
            # Every (i, i^half) pair is unique → no gather aliasing.
            partner_idx = idx ^ half
            x_partner = tl.load(base + partner_idx)

            # Butterfly: is_upper when bit `stage` of idx is 0.
            is_upper = (idx & half) == 0
            x = tl.where(is_upper,
                         x + x_partner,    # upper a → a + b
                         x_partner - x)    # lower b → a − b

        # 1/√D normalization: makes (H/√D) an isometry, preserving dot products.
        x = x * SCALE
        tl.store(base + idx, x)


# ── Internal dispatcher ───────────────────────────────────────────────────

def _wht_rotate(x: torch.Tensor) -> torch.Tensor:
    """
    Apply normalized WHT to the last dimension.

    Routes to the Triton kernel (including TRITON_INTERPRET=1 mode) when
    Triton is available; falls back to _wht_torch otherwise.

    Args:
        x: (..., HEAD_DIM) — any leading batch dims, must be last-dim HEAD_DIM
    Returns:
        New tensor with same shape, dtype, and device; WHT applied to last dim.
        Input is never modified.
    """
    if x.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"WHT rotation requires last dim = {HEAD_DIM} (head_dim), "
            f"got {x.shape[-1]}. Check OmniConfig.head_dim."
        )

    if not _HAS_TRITON:
        return _wht_torch(x)

    orig_shape = x.shape
    # Flatten to (N_rows, HEAD_DIM); clone so the in-place kernel doesn't touch x
    x_out = x.contiguous().clone().reshape(-1, HEAD_DIM)
    n_rows = x_out.shape[0]

    _wht_kernel[(n_rows,)](
        x_out,
        HEAD_DIM,           # stride_row: HEAD_DIM for a contiguous 2-D tensor
        n_rows,
        HEAD_DIM=HEAD_DIM,
        LOG2_HD=LOG2_HEAD_DIM,
        SCALE=_SCALE,
    )

    return x_out.reshape(orig_shape)


# ── Public API ────────────────────────────────────────────────────────────

def rotate_queries(q: torch.Tensor) -> torch.Tensor:
    """
    Hadamard-rotate query projections (called once per decode step).

    MANIFEST Invariant #1: this is the ONLY place WHT touches Q.
    Never call WHT inside the attention inner loop.

    After rotation, the full attention computation is unchanged:
        WHT(q) @ WHT(k)^T / √d  ==  q @ k^T / √d   (for all q, k)

    Args:
        q: (B, n_heads, T, head_dim)
    Returns:
        (B, n_heads, T, head_dim) — rotated; same shape, dtype, device
    """
    return _wht_rotate(q)


def rotate_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hadamard-rotate key and value tensors (called once at prefill).

    K and V are stored permanently in rotated space for the lifetime of the
    KV cache. They are NEVER re-rotated during decode.

    Rotating V is necessary for INT4 quantization quality:
    the WHT sphericizes the value distribution, so quantization noise
    is isotropic and Lloyd-Max centroids achieve lower MSE.

    Args:
        k: (B, n_kv_heads, S, head_dim)
        v: (B, n_kv_heads, S, head_dim)
    Returns:
        (k_rot, v_rot): same shapes, dtype, device as k, v
    """
    return _wht_rotate(k), _wht_rotate(v)
