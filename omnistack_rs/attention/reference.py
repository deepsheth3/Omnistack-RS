"""
OmniStack-RS — Stage 6: GQA Reference Attention (Numerical Anchor)

Pure-PyTorch Grouped-Query Attention with tile-by-tile online softmax.
This module is the ground-truth reference for validating the fused Triton
kernel in Phase 4. The online algorithm here must match the Triton inner
loop exactly — same update rule, same accumulator semantics.

Strict constraints (Architect's instruction):
  - NO F.scaled_dot_product_attention — write every step explicitly
  - Softmax ALWAYS in float32 (MANIFEST Invariant #6)
  - W_eff = W_base + B @ A * (α/r) for Shadow LoRA merge

Online softmax algorithm (Milakov & Gimelshein 2018 / Dao et al. 2022):

  For each tile of K/V keys (indices s_start..s_end):

    s_tile = Q @ K_tile^T * scale              # (B, H, T, TILE_S) — never (T, S)
    m_new  = max(m_old, max(s_tile))           # update running max
    alpha  = exp(m_old - m_new)               # rescale factor for old accumulator
    p      = exp(s_tile - m_new)              # (B, H, T, TILE_S) — unnorm weights

    l = alpha * l + sum(p)                    # running denominator (normalizer)
    O = alpha * O + p @ V_tile                # running weighted sum

  out = O / l                                 # normalize once at the end

Invariants:
  - m is monotonically non-decreasing across tiles (proved by max update rule)
  - O / l converges to the exact softmax output regardless of tile order
  - The tile score buffer (T × TILE_S) is the only O(seq_len) allocation;
    the full (T × S) score matrix is NEVER materialized
  - exp() is never called on unshifted scores (overflow impossible)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from omnistack_rs.config import OmniConfig


# ── GQA utility ───────────────────────────────────────────────────────────

def repeat_kv(x: torch.Tensor, n_groups: int) -> torch.Tensor:
    """
    Expand KV heads to match query heads for GQA (no data copy).

    Args:
        x:        (B, n_kv_heads, S, head_dim)
        n_groups: n_heads // n_kv_heads

    Returns:
        (B, n_heads, S, head_dim) — each KV head tiled n_groups times
    """
    if n_groups == 1:
        return x
    B, n_kv, S, d = x.shape
    return (
        x.unsqueeze(2)
         .expand(B, n_kv, n_groups, S, d)
         .reshape(B, n_kv * n_groups, S, d)
    )


# ── Core attention function ────────────────────────────────────────────────

def reference_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    tile_size: int = 128,
) -> torch.Tensor:
    """
    Manual GQA attention with tile-by-tile online softmax — Numerical Anchor.

    The online algorithm processes keys in tiles of `tile_size` columns,
    maintaining running state (m, l, O) that fits in registers. The full
    (T, S) score matrix is NEVER materialized; working memory per tile is
    O(T × tile_size), constant in S.

    This is the exact algorithm the Phase 4 Triton kernel implements; the
    only difference is that the Triton version uses SRAM tiles and TMA loads.

    Args:
        q:         (B, n_heads,    T, head_dim)
        k:         (B, n_kv_heads, S, head_dim)
        v:         (B, n_kv_heads, S, head_dim)
        mask:      additive bias broadcastable to (B, n_heads, T, S)
                   0.0 = attend, float('-inf') = causal mask / padding
        scale:     QK scale; defaults to 1/sqrt(head_dim)
        tile_size: number of key columns processed per tile (default 128)
                   smaller values stress-test the running-state update rule

    Returns:
        (B, n_heads, T, head_dim), dtype matches q
    """
    n_heads    = q.shape[1]
    n_kv_heads = k.shape[1]
    head_dim   = q.shape[-1]
    n_groups   = n_heads // n_kv_heads
    S          = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # GQA: expand K, V so every Q head has a dedicated KV head
    k = repeat_kv(k, n_groups)   # (B, n_heads, S, head_dim)
    v = repeat_kv(v, n_groups)

    # ── Upcast to FP32 ────────────────────────────────────────────────────
    # BF16 has 7 mantissa bits; exp() on uncast scores risks underflow/overflow.
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    B = q.shape[0]
    T = q.shape[2]

    # ── Running-state initialization ──────────────────────────────────────
    # m: running max of scores seen so far (per query position)
    # l: running unnormalized denominator (sum of exp(score - m))
    # O: running weighted value accumulator (sum of exp(score - m) * v)
    #
    # All three are O(T × head_dim) — independent of total sequence length S.
    m = q_f.new_full((B, n_heads, T, 1), float("-inf"))   # (B, H, T, 1)
    l = q_f.new_zeros(B, n_heads, T, 1)                    # (B, H, T, 1)
    O = q_f.new_zeros(B, n_heads, T, head_dim)             # (B, H, T, head_dim)

    # ── Tile loop ─────────────────────────────────────────────────────────
    # Each iteration handles TILE_S key/value positions.
    # The score buffer (B, H, T, TILE_S) is the ONLY per-tile allocation;
    # it is overwritten every iteration, never grown.
    for s_start in range(0, S, tile_size):
        s_end  = min(s_start + tile_size, S)

        k_tile = k_f[:, :, s_start:s_end, :]   # (B, H, TILE, head_dim)
        v_tile = v_f[:, :, s_start:s_end, :]   # (B, H, TILE, head_dim)

        # Score tile: (B, H, T, TILE) — the only score buffer
        s_tile = torch.matmul(q_f, k_tile.transpose(-2, -1)) * scale

        if mask is not None:
            s_tile = s_tile + mask[..., s_start:s_end].float()

        # ── Running max update ────────────────────────────────────────────
        # m is non-decreasing: max(m_old, tile_max) ≥ m_old always.
        m_tile = s_tile.amax(dim=-1, keepdim=True)   # (B, H, T, 1)
        m_new  = torch.maximum(m, m_tile)             # (B, H, T, 1)

        # ── Rescale old accumulators to the new max ───────────────────────
        # Because O was accumulated under m_old, each term in O is
        #   exp(score_k - m_old) * v_k.
        # After shifting to m_new, those terms become
        #   exp(score_k - m_new) * v_k = exp(m_old - m_new) * (old term).
        alpha = torch.exp(m - m_new)                  # (B, H, T, 1)

        # Unnormalized weights for this tile (never store the full (T,S) version)
        p = torch.exp(s_tile - m_new)                 # (B, H, T, TILE)

        # ── Accumulator update ────────────────────────────────────────────
        l = alpha * l + p.sum(dim=-1, keepdim=True)   # (B, H, T, 1)
        O = alpha * O + torch.matmul(p, v_tile)        # (B, H, T, head_dim)
        m = m_new

    # ── Final normalization ───────────────────────────────────────────────
    # O holds sum_k exp(s_k - m_final) * v_k; l holds sum_k exp(s_k - m_final).
    # Dividing gives the exact softmax-weighted value: sum_k softmax(s_k) * v_k.
    out_f = O / l   # (B, H, T, head_dim) — all FP32

    return out_f.to(q.dtype)


# ── Shadow LoRA merge helper ───────────────────────────────────────────────

def _lora_delta(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    alpha: float,
    rank: int,
) -> torch.Tensor:
    """
    Compute the LoRA weight delta: ΔW = B @ A * (α / r).

    Args:
        lora_A: (r, in_features)      — down-projection (trained)
        lora_B: (out_features, r)     — up-projection (trained)
        alpha:  LoRA scaling factor α
        rank:   LoRA rank r

    Returns:
        (out_features, in_features) — same shape as the base weight matrix
    """
    return (lora_B @ lora_A) * (alpha / rank)


# ── ReferenceAttention module ──────────────────────────────────────────────

class ReferenceAttention(nn.Module):
    """
    GQA attention block: ground-truth reference path (Master + Shadow).

    Roles:
      1. Numerical anchor — validates Phase 4 fused Triton kernel (atol=5e-2)
      2. Shadow LoRA demo — shows W_eff = W_base + B @ A * (α/r) merge
      3. End-to-end perplexity baseline without quantization noise

    LoRA is applied to Q and V projections only (MANIFEST §Phase 5).
    K and O projections are frozen (shared across users in Master LLM).
    """

    def __init__(self, config: OmniConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        q_out = config.n_heads * config.head_dim
        kv_out = config.n_kv_heads * config.head_dim

        self.q_proj = nn.Linear(d, q_out, bias=False)
        self.k_proj = nn.Linear(d, kv_out, bias=False)
        self.v_proj = nn.Linear(d, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, d, bias=False)

        self._scale = 1.0 / math.sqrt(config.head_dim)

    # ── Shadow LoRA weight merge ──────────────────────────────────────────

    def apply_lora(
        self,
        proj: str,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        alpha: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> None:
        """
        Merge Shadow LoRA weights into a projection in-place.

        W_eff = W_base + B @ A * (α / r)

        After this call, the projection permanently uses the merged weights.
        Call once per inference session (or use effective_weight() for read-only).

        Args:
            proj:   "q_proj" | "k_proj" | "v_proj" | "o_proj"
            lora_A: (r, in_features)   — Shadow down-projection
            lora_B: (out_features, r)  — Shadow up-projection
            alpha:  override for config.lora_alpha
            rank:   override for config.lora_rank
        """
        if not hasattr(self, proj) or not isinstance(getattr(self, proj), nn.Linear):
            raise ValueError(
                f"Unknown projection '{proj}'. Choose from: q_proj, k_proj, v_proj, o_proj"
            )
        α = alpha if alpha is not None else self.config.lora_alpha
        r = rank if rank is not None else self.config.lora_rank
        layer: nn.Linear = getattr(self, proj)

        delta = _lora_delta(lora_A, lora_B, α, r).to(layer.weight.dtype)
        with torch.no_grad():
            layer.weight.add_(delta)

    def effective_weight(
        self,
        proj: str,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        alpha: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Return W_eff = W_base + B @ A * (α / r) without modifying state.

        Useful for inspecting the merge formula in tests without side-effects.
        """
        α = alpha if alpha is not None else self.config.lora_alpha
        r = rank if rank is not None else self.config.lora_rank
        layer: nn.Linear = getattr(self, proj)
        delta = _lora_delta(lora_A, lora_B, α, r)
        return layer.weight + delta.to(layer.weight.dtype)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        GQA forward pass with tile-by-tile online softmax.

        Args:
            x:    (B, T, hidden_dim)
            mask: additive attention bias, broadcastable to (B, 1, T, T)
                  Use causal_mask() to generate a standard causal mask.

        Returns:
            (B, T, hidden_dim)
        """
        B, T, _ = x.shape
        H = self.config.n_heads
        Hkv = self.config.n_kv_heads
        d = self.config.head_dim

        q = self.q_proj(x).reshape(B, T, H,   d).transpose(1, 2)   # (B, H,   T, d)
        k = self.k_proj(x).reshape(B, T, Hkv, d).transpose(1, 2)   # (B, Hkv, T, d)
        v = self.v_proj(x).reshape(B, T, Hkv, d).transpose(1, 2)   # (B, Hkv, T, d)

        attn_out = reference_attn(q, k, v, mask=mask, scale=self._scale)

        out = attn_out.transpose(1, 2).reshape(B, T, H * d)         # (B, T, H*d)
        return self.o_proj(out)                                       # (B, T, hidden_dim)

    def causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Standard autoregressive causal mask: (1, 1, T, T).
        Upper triangle is -inf (future tokens masked), lower triangle is 0.
        """
        mask = torch.zeros(1, 1, T, T, device=device)
        upper = torch.ones(T, T, device=device, dtype=torch.bool).triu(diagonal=1)
        return mask.masked_fill(upper, float("-inf"))
