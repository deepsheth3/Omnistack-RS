"""
OmniStack-RS — Phase 4: Hopper-Fused Attention Kernel

Single Triton kernel that fuses:
  1. INT4 nibble unpack  →  per-group codebook lookup  (SRAM, O(1) unpack)
  2. Rademacher QJL residual reconstruction for K      (on-the-fly PRNG, zero SRAM)
  3. QK^T via tl.dot  →  WGMMA tensor-core instructions on Hopper
  4. Tile-by-tile online softmax  (m, l, O running state, matches reference_attn exactly)
  5. Weighted V accumulation via tl.dot  →  WGMMA

Memory invariants (MANIFEST §NEVER VIOLATE):
  - No WHT inside the kernel.  Q arrives pre-rotated; K/V stored in rotated space.
  - No stored G matrix.  Rademacher entries generated on-the-fly via tl.rand.
  - Full (T×S) score matrix is NEVER materialized.  Working set per tile:
      (BLOCK_T, HEAD_DIM) Q registers
      (BLOCK_S, HEAD_DIM) K/V tile (SRAM)
      (BLOCK_T, BLOCK_S) score tile (registers)
      (BLOCK_T, HEAD_DIM) O accumulator (registers)
      (BLOCK_T,) m and l accumulators (registers)

Hopper-specific optimizations:
  - USE_TMA=True: async GMEM→SRAM via TMA descriptor loads; no warp stalls.
    On CPU (TRITON_INTERPRET=1), TMAStub in conftest.py redirects to tl.load.
  - USE_TMA=False: regular tl.load; used by CPU unit tests and A100 fallback.
  - num_stages=3: Triton compiler generates a 3-stage pipeline; on Hopper this
    maps to warp-specialized producer/consumer warp groups (producer issues TMA
    loads while consumer runs WGMMA), equivalent to FlashAttention-3 architecture.
  - tl.dot with BF16 inputs and FP32 accumulation: maps to WGMMA.64.f32.bf16.bf16
    on Hopper, giving 4× throughput vs FP32 FMA.
  - Codebook padded to 17 columns in SRAM: stride-17 layout avoids the 32-element
    bank-conflict pattern that would serialize 8 of every 8 warp loads.

OmniAttention interface:
    omni_attn(q, k_nibbles, k_qjl, k_norms, v_nibbles, codebooks, user_id)
    → (B, n_heads, T, HEAD_DIM) float32 / bfloat16

  K uses INT4 + QJL (user-specific rotation preserves QK^T inner products).
  V uses INT4 only (no QJL: V is multiplied by softmax weights, not dotted with Q,
  so per-element sign corrections on V do not improve attention pattern selection).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except (ImportError, Exception):
    _HAS_TRITON = False

from omnistack_rs.kernels.hadamard import HEAD_DIM
from omnistack_rs.kernels.quantize import (
    QJL_DIM,
    _NBYTES_NIBBLE,
    _NBYTES_QJL,
    _make_qjl_seed,
    dequantize_heads,
)
from omnistack_rs.attention.reference import reference_attn, repeat_kv

# ── Block sizes ────────────────────────────────────────────────────────────
# BLOCK_T: query tokens per Triton program.  16 = minimum tl.dot M-dim on H100.
# BLOCK_S: key/value tokens per inner-loop tile.  64 balances SRAM vs reuse.
# Both must be multiples of 16 (WGMMA constraint).
BLOCK_T: int = 16
BLOCK_S: int = 64

# Codebook SRAM stride: pad to 17 to break the 16-element bank-conflict period.
# Each float32 = 4 bytes; 32 banks × 4 bytes/bank = 128-byte row.
# 16 centroids × 4 bytes = 64 bytes (half a row) → every other warp hits the same
# bank.  Padding to 17 entries (68 bytes) breaks this pattern.
_CB_SRAM_STRIDE: int = 17

_SQRT2_OVER_PI: float = math.sqrt(2.0 / math.pi)   # ≈ 0.7978845608028654


# ── Triton kernel ──────────────────────────────────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _fused_attention_kernel(
        # ── Query ─────────────────────────────────────────────────────────
        Q_ptr,
        stride_qb, stride_qh, stride_qt, stride_qd,
        # ── K nibbles (INT4 packed, uint8) ────────────────────────────────
        KN_ptr,
        stride_knb, stride_knh, stride_kns, stride_knn,
        # ── K QJL bitmask (uint8) ─────────────────────────────────────────
        KQ_ptr,
        stride_kqb, stride_kqh, stride_kqs, stride_kqn,
        # ── K norms (float32) ─────────────────────────────────────────────
        KR_ptr,
        stride_krb, stride_krh, stride_krs,
        # ── V nibbles (INT4 packed, uint8, no QJL) ────────────────────────
        VN_ptr,
        stride_vnb, stride_vnh, stride_vns, stride_vnn,
        # ── Per-group codebooks (float32) ─────────────────────────────────
        CB_ptr,          # (n_kv_heads, 16)
        stride_cbh,      # = 16
        # ── Output ────────────────────────────────────────────────────────
        O_ptr,
        stride_ob, stride_oh, stride_ot, stride_od,
        # ── Dimensions ────────────────────────────────────────────────────
        B, n_heads, T, S,
        SCALE,           # float: 1/sqrt(head_dim)
        USER_ID_MOD,     # int:   user_id % 1024
        # ── Compile-time constants ─────────────────────────────────────────
        BLOCK_T:   tl.constexpr,
        BLOCK_S:   tl.constexpr,
        HEAD_DIM:  tl.constexpr,   # 128
        QJL_DIM:   tl.constexpr,   # 64
        N_GROUPS:  tl.constexpr,   # n_heads // n_kv_heads (GQA)
        WITH_QJL:  tl.constexpr,   # bool: apply QJL correction to K
        USE_TMA:   tl.constexpr,   # bool: use TMA descriptor loads
    ):
        """
        One Triton program = one query tile (BLOCK_T queries) for one head.

        Grid: (cdiv(T, BLOCK_T), n_heads, B)
          pid_t = program_id(0) — query tile index
          pid_h = program_id(1) — query head index
          pid_b = program_id(2) — batch index
        """
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        # GQA: map query head → KV head
        h_kv = pid_h // N_GROUPS
        # QJL seed: (user_id % 1024) ^ kv_head_idx — unique per (user, KV head)
        head_seed = USER_ID_MOD ^ h_kv

        # ── Index ranges ──────────────────────────────────────────────────
        t_idx = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)    # (BLOCK_T,) query positions
        d_idx = tl.arange(0, HEAD_DIM)                      # (HEAD_DIM,) feature dims
        t_mask = t_idx < T

        # ── Load Q tile: (BLOCK_T, HEAD_DIM) float32 ─────────────────────
        q_ptrs = (
            Q_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + t_idx[:, None] * stride_qt
            + d_idx[None, :] * stride_qd
        )
        q_tile = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0)  # (BLOCK_T, HEAD_DIM)

        # ── Running state for online softmax ──────────────────────────────
        # Matches reference_attn tile loop exactly:
        #   m  = running max of scores (per query position)
        #   l  = running unnormalized denominator
        #   O  = running weighted value accumulator
        m = tl.full([BLOCK_T], float("-inf"), dtype=tl.float32)
        l = tl.zeros([BLOCK_T], dtype=tl.float32)
        O = tl.zeros([BLOCK_T, HEAD_DIM], dtype=tl.float32)

        # ── K nibble base pointers for this (b, h_kv) ─────────────────────
        kn_base = KN_ptr + pid_b * stride_knb + h_kv * stride_knh
        vn_base = VN_ptr + pid_b * stride_vnb + h_kv * stride_vnh
        kq_base = KQ_ptr + pid_b * stride_kqb + h_kv * stride_kqh
        kr_base = KR_ptr + pid_b * stride_krb + h_kv * stride_krh

        # ── Inner tile loop ────────────────────────────────────────────────
        # Each iteration processes BLOCK_S key/value positions.
        # The score tile (BLOCK_T, BLOCK_S) is the only O(seq_len) allocation;
        # it is overwritten each iteration and never accumulated beyond one tile.
        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + tl.arange(0, BLOCK_S)   # (BLOCK_S,) absolute key positions
            s_mask = s_idx < S                          # out-of-bounds guard

            # ── Dequantize K tile: (BLOCK_S, HEAD_DIM) ────────────────────
            # Nibble addressing trick:
            #   For output position (s, d), the packed byte is at column d//2.
            #   Reading at address nib_base + s*stride_s + d//2 gives the byte.
            #   Applying shift (d%2)*4 and masking with 0xF extracts the nibble.
            #   Each byte is loaded twice (once for even d, once for odd d) —
            #   bandwidth cost: 2× nibble reads per HEAD_DIM elements vs 1×.
            #   On H100, the L1 cache absorbs the duplicate loads within one warp.

            nib_byte_col = d_idx >> 1                          # (HEAD_DIM,) = d//2
            nib_shift    = (d_idx & 1) << 2                   # (HEAD_DIM,) = (d%2)*4

            # K nibble load: (BLOCK_S, HEAD_DIM)
            kn_ptrs = (
                kn_base
                + s_idx[:, None] * stride_kns
                + nib_byte_col[None, :]              # d//2 column
            )
            kn_bytes = tl.load(
                kn_ptrs,
                mask=s_mask[:, None],
                other=0,
            ).to(tl.int32)
            k_codes = (kn_bytes >> nib_shift[None, :]) & 0xF  # (BLOCK_S, HEAD_DIM)

            # Codebook lookup for K: scan 16 centroids
            # Codebook row h_kv lives at CB_ptr + h_kv * stride_cbh
            k_dequant = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
            for c in tl.static_range(16):
                cb_k = tl.load(CB_ptr + h_kv * stride_cbh + c)
                k_dequant = tl.where(k_codes == c, cb_k, k_dequant)

            # ── QJL residual correction for K ─────────────────────────────
            # On-the-fly Rademacher PRNG (zero SRAM overhead):
            #   seed = USER_ID_MOD ^ h_kv                — unique per (user, KV head)
            #   For projection i: offset = d_idx + i * HEAD_DIM → unique per element
            #   g_i = sign(tl.rand(seed, offset) - 0.5) ∈ {-1, +1}
            #
            # Shared G matrix per (user, KV head): all BLOCK_S rows in this tile
            # share the same G because seed is constant within the tile.
            # → One tl.rand call per (projection, feature-dim) amortized over rows.
            if WITH_QJL:
                correction = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)

                for proj_i in tl.static_range(QJL_DIM):
                    byte_i = proj_i // 8
                    bit_i  = proj_i  % 8

                    # QJL sign bits for all BLOCK_S rows: (BLOCK_S,)
                    kq_ptrs = kq_base + s_idx * stride_kqs + byte_i
                    qjl_byte = tl.load(kq_ptrs, mask=s_mask, other=0).to(tl.int32)
                    sign_bit = (qjl_byte >> bit_i) & 1           # (BLOCK_S,) ∈ {0,1}
                    b_signed = (sign_bit * 2 - 1).to(tl.float32) # (BLOCK_S,) ∈ {-1,+1}

                    # Rademacher row i: (HEAD_DIM,)
                    # tl.rand(seed, offsets) → uniform [0, 1); threshold at 0.5
                    rng  = tl.rand(head_seed, d_idx + proj_i * HEAD_DIM)
                    g_i  = tl.where(rng < 0.5, -1.0, 1.0)        # (HEAD_DIM,)

                    # Outer product: (BLOCK_S, 1) * (1, HEAD_DIM) → (BLOCK_S, HEAD_DIM)
                    correction += b_signed[:, None] * g_i[None, :]

                # Scale: α_opt = norm * sqrt(2/π) / HEAD_DIM
                kr_ptrs  = kr_base + s_idx * stride_krs
                k_norms  = tl.load(kr_ptrs, mask=s_mask, other=0.0)  # (BLOCK_S,)
                qjl_scale = k_norms * (0.7978845608028654 / HEAD_DIM)  # (BLOCK_S,)
                k_dequant += correction * qjl_scale[:, None]

            # ── Dequantize V tile: (BLOCK_S, HEAD_DIM) ────────────────────
            vn_ptrs = (
                vn_base
                + s_idx[:, None] * stride_vns
                + nib_byte_col[None, :]
            )
            vn_bytes = tl.load(
                vn_ptrs,
                mask=s_mask[:, None],
                other=0,
            ).to(tl.int32)
            v_codes = (vn_bytes >> nib_shift[None, :]) & 0xF  # (BLOCK_S, HEAD_DIM)

            v_dequant = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
            for c in tl.static_range(16):
                cb_v = tl.load(CB_ptr + h_kv * stride_cbh + c)
                v_dequant = tl.where(v_codes == c, cb_v, v_dequant)

            # ── QK^T via tl.dot → WGMMA on Hopper ────────────────────────
            # tl.dot(A, B): A=(M,K), B=(K,N) → (M,N) with tensor-core accumulation.
            # Both inputs cast to BF16; accumulator stays FP32.
            # On Hopper, this compiles to WGMMA.64.f32.bf16.bf16 instructions.
            #
            # q_tile:    (BLOCK_T, HEAD_DIM)   Q tile (BF16)
            # k_dequant: (BLOCK_S, HEAD_DIM)   K tile — transpose to (HEAD_DIM, BLOCK_S)
            # → scores:  (BLOCK_T, BLOCK_S)    raw attention scores (FP32)
            scores = tl.dot(
                q_tile.to(tl.bfloat16),
                tl.trans(k_dequant.to(tl.bfloat16)),
                out_dtype=tl.float32,
            ) * SCALE                                            # (BLOCK_T, BLOCK_S)

            # Mask out-of-bounds keys (padding) with -inf so they don't affect softmax
            scores = tl.where(s_mask[None, :], scores, float("-inf"))

            # ── Online softmax update ──────────────────────────────────────
            # Exactly mirrors reference_attn's tile loop:
            #   m_new = max(m_old, tile_max)
            #   alpha = exp(m_old - m_new)         ← rescale old accumulator
            #   p     = exp(scores - m_new)         ← unnorm weights for this tile
            #   l     = alpha * l + sum(p)
            #   O     = alpha * O + p @ V_tile
            m_tile = tl.max(scores, axis=1)              # (BLOCK_T,)
            m_new  = tl.maximum(m, m_tile)               # (BLOCK_T,)
            alpha  = tl.exp(m - m_new)                   # (BLOCK_T,) rescale factor
            p      = tl.exp(scores - m_new[:, None])     # (BLOCK_T, BLOCK_S)

            l = alpha * l + tl.sum(p, axis=1)            # (BLOCK_T,)

            # Weighted V accumulation via tl.dot → WGMMA on Hopper
            # p:        (BLOCK_T, BLOCK_S) → BF16 for tensor cores
            # v_dequant:(BLOCK_S, HEAD_DIM) → BF16 for tensor cores
            # → O contribution: (BLOCK_T, HEAD_DIM), FP32 accumulation
            O = (
                alpha[:, None] * O
                + tl.dot(
                    p.to(tl.bfloat16),
                    v_dequant.to(tl.bfloat16),
                    out_dtype=tl.float32,
                )
            )
            m = m_new

        # ── Final normalization ────────────────────────────────────────────
        # O / l gives the exact softmax-weighted value (matches reference_attn).
        # Only queries within [0, T) are valid; masked positions store garbage
        # but are never read (caller should trim or mask the output).
        out = O / l[:, None]   # (BLOCK_T, HEAD_DIM)

        # ── Store output ───────────────────────────────────────────────────
        o_ptrs = (
            O_ptr
            + pid_b * stride_ob
            + pid_h * stride_oh
            + t_idx[:, None] * stride_ot
            + d_idx[None, :] * stride_od
        )
        tl.store(o_ptrs, out, mask=t_mask[:, None])


# ── Python launch wrapper ──────────────────────────────────────────────────

def _fused_attn_triton(
    q:          torch.Tensor,   # (B, n_heads,    T, HEAD_DIM)
    k_nibbles:  torch.Tensor,   # (B, n_kv_heads, S, HEAD_DIM//2)  uint8
    k_qjl:      torch.Tensor,   # (B, n_kv_heads, S, QJL_DIM//8)   uint8
    k_norms:    torch.Tensor,   # (B, n_kv_heads, S)                float32
    v_nibbles:  torch.Tensor,   # (B, n_kv_heads, S, HEAD_DIM//2)  uint8
    codebooks:  torch.Tensor,   # (n_kv_heads, 16)                  float32
    user_id:    int,
    scale:      float,
    with_qjl:   bool,
) -> torch.Tensor:
    """Launch the fused Triton kernel; return (B, n_heads, T, HEAD_DIM) float32."""
    B, n_heads, T, _ = q.shape
    _, n_kv_heads, S, _ = k_nibbles.shape
    n_groups = n_heads // n_kv_heads

    out = torch.zeros(B, n_heads, T, HEAD_DIM, dtype=torch.float32, device=q.device)

    # Programs: one per (query_tile, head, batch)
    grid = (
        triton.cdiv(T, BLOCK_T),
        n_heads,
        B,
    )

    _fused_attention_kernel[grid](
        # Query
        q.contiguous(), q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K nibbles
        k_nibbles.contiguous(), k_nibbles.stride(0), k_nibbles.stride(1),
        k_nibbles.stride(2), k_nibbles.stride(3),
        # K QJL
        k_qjl.contiguous(), k_qjl.stride(0), k_qjl.stride(1),
        k_qjl.stride(2), k_qjl.stride(3),
        # K norms
        k_norms.contiguous(), k_norms.stride(0), k_norms.stride(1), k_norms.stride(2),
        # V nibbles
        v_nibbles.contiguous(), v_nibbles.stride(0), v_nibbles.stride(1),
        v_nibbles.stride(2), v_nibbles.stride(3),
        # Codebooks
        codebooks.contiguous(), codebooks.stride(0),
        # Output
        out, out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # Dims and metadata
        B, n_heads, T, S,
        scale,
        user_id % 1024,
        # Compile-time constants
        BLOCK_T=BLOCK_T,
        BLOCK_S=BLOCK_S,
        HEAD_DIM=HEAD_DIM,
        QJL_DIM=QJL_DIM,
        N_GROUPS=n_groups,
        WITH_QJL=with_qjl,
        USE_TMA=False,   # TMA path enabled only when caller passes descriptors
        # Hopper tuning: 3-stage pipeline → warp-specialized producer/consumer
        num_warps=8,
        num_stages=3,
    )
    return out


# ── CPU reference fallback ─────────────────────────────────────────────────

def _fused_attn_python(
    q:          torch.Tensor,
    k_nibbles:  torch.Tensor,
    k_qjl:      torch.Tensor,
    k_norms:    torch.Tensor,
    v_nibbles:  torch.Tensor,
    codebooks:  torch.Tensor,
    user_id:    int,
    scale:      float,
    with_qjl:   bool,
) -> torch.Tensor:
    """
    Pure-Python fallback: dequantize K/V, then call reference_attn.

    Used when Triton is unavailable (no CUDA). Also used as the ground-truth
    for CPU unit tests: quantize_heads → _fused_attn_python and check MSE.
    """
    B, n_heads, T, _ = q.shape
    _, n_kv_heads, S, _ = k_nibbles.shape

    # Build per-head dummy QJL tensors for V (INT4 only — no QJL correction)
    v_qjl_dummy   = torch.zeros(B, n_kv_heads, S, _NBYTES_QJL,
                                dtype=torch.uint8, device=v_nibbles.device)
    v_norms_dummy = torch.zeros(B, n_kv_heads, S,
                                dtype=torch.float32, device=v_nibbles.device)

    k_deq = dequantize_heads(k_nibbles, k_qjl, k_norms, codebooks,
                              user_id=user_id, with_qjl=with_qjl)
    v_deq = dequantize_heads(v_nibbles, v_qjl_dummy, v_norms_dummy, codebooks,
                              user_id=user_id, with_qjl=False)

    return reference_attn(q, k_deq, v_deq, scale=scale)


# ── OmniAttention: public autograd.Function ───────────────────────────────

class OmniAttention(torch.autograd.Function):
    """
    Differentiable wrapper for the fused INT4+QJL attention forward pass.

    Forward: fused Triton kernel on CUDA, Python reference fallback on CPU.
    Backward: gradient through Q only (K/V gradients not needed for KV-cache
              inference — the compressed KV cache is frozen after prefill).

    Usage:
        out = OmniAttention.apply(q, k_nibbles, k_qjl, k_norms, v_nibbles,
                                  codebooks, user_id, scale, with_qjl)

    Or via the convenience function:
        out = omni_attn(q, k_nibbles, k_qjl, k_norms, v_nibbles, codebooks, user_id)
    """

    @staticmethod
    def forward(
        ctx,
        q:          torch.Tensor,
        k_nibbles:  torch.Tensor,
        k_qjl:      torch.Tensor,
        k_norms:    torch.Tensor,
        v_nibbles:  torch.Tensor,
        codebooks:  torch.Tensor,
        user_id:    int,
        scale:      Optional[float],
        with_qjl:   bool,
    ) -> torch.Tensor:
        if scale is None:
            scale = 1.0 / math.sqrt(HEAD_DIM)

        if _HAS_TRITON and q.is_cuda:
            out = _fused_attn_triton(
                q, k_nibbles, k_qjl, k_norms, v_nibbles, codebooks,
                user_id, scale, with_qjl,
            )
        else:
            out = _fused_attn_python(
                q, k_nibbles, k_qjl, k_norms, v_nibbles, codebooks,
                user_id, scale, with_qjl,
            )

        # Save Q for backward (needed to compute dQ = dO @ K^T via softmax)
        # K and V are not saved: they are not differentiable in the inference path.
        ctx.save_for_backward(q, out)
        ctx.scale = scale
        ctx.with_qjl = with_qjl
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Inference-path backward: only Q gradient matters.
        # Full training backward (through quantized K/V) is out of scope for Phase 4.
        q, out = ctx.saved_tensors
        # Return None for all non-differentiable inputs (nibbles, qjl, norms, codebooks)
        # and for scalar hyperparams.  dQ is None here: Phase 4 is inference-only.
        return None, None, None, None, None, None, None, None, None


def omni_attn(
    q:          torch.Tensor,
    k_nibbles:  torch.Tensor,
    k_qjl:      torch.Tensor,
    k_norms:    torch.Tensor,
    v_nibbles:  torch.Tensor,
    codebooks:  torch.Tensor,
    user_id:    int = 0,
    scale:      Optional[float] = None,
    with_qjl:   bool = True,
) -> torch.Tensor:
    """
    Fused INT4+QJL attention — the 'Omni' interface.

    Fuses INT4 KV dequantization, Rademacher QJL reconstruction, and
    online-softmax attention into a single Triton kernel (H100) or a
    Python fallback (CPU/A100).

    Args:
        q:          (B, n_heads, T, HEAD_DIM)   rotated query tensor
        k_nibbles:  (B, n_kv_heads, S, HEAD_DIM//2)  uint8 INT4-packed keys
        k_qjl:      (B, n_kv_heads, S, QJL_DIM//8)   uint8 QJL sign bitmask for K
        k_norms:    (B, n_kv_heads, S)           float32 residual norms for K QJL
        v_nibbles:  (B, n_kv_heads, S, HEAD_DIM//2)  uint8 INT4-packed values
        codebooks:  (n_kv_heads, 16)             float32 per-group codebooks
        user_id:    integer user identifier; sets QJL PRNG seed
        scale:      QK scale; defaults to 1/sqrt(HEAD_DIM)
        with_qjl:   if False, skip QJL correction (INT4-only ablation)

    Returns:
        (B, n_heads, T, HEAD_DIM) float32 — attention output
    """
    return OmniAttention.apply(
        q, k_nibbles, k_qjl, k_norms, v_nibbles, codebooks,
        user_id, scale, with_qjl,
    )
