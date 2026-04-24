"""
OmniStack-RS — Stage 6: GQA Reference Attention (Numerical Anchor)

Pure-PyTorch Grouped-Query Attention with manual FP32 softmax.
This module is the ground-truth reference path for validating the
fused Triton kernel in Phase 4.

Strict constraints (Architect's instruction):
  - NO F.scaled_dot_product_attention — write every step explicitly
  - Softmax ALWAYS in float32 (MANIFEST Invariant #6)
  - W_eff = W_base + B @ A * (α/r) for Shadow LoRA merge

Mathematical flow:
  1. Q, K, V projections (with optional LoRA merge)
  2. GQA head expansion: repeat_kv(K, n_groups), repeat_kv(V, n_groups)
  3. S = Q @ K^T / sqrt(head_dim)          [FP32]
  4. S = S - max(S, dim=-1)                [online softmax stability]
  5. W = exp(S) / sum(exp(S), dim=-1)      [FP32 throughout]
  6. O = W @ V                             [FP32, cast back to input dtype]
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
) -> torch.Tensor:
    """
    Manual GQA attention — the Numerical Anchor for Phase 4 kernel validation.

    Every precision decision is explicit so the fused Triton kernel can be
    validated step-by-step against this reference (atol=5e-2 on H100).

    Steps mirror the fused kernel's inner loop exactly:
      1. Upcast Q/K/V to float32
      2. Scaled dot-product: S = Q @ K^T * scale
      3. Additive mask (0=attend, -inf=mask)
      4. Online softmax: subtract row max → exp() → normalize  (all FP32)
      5. Weighted sum: O = softmax(S) @ V
      6. Downcast output back to input dtype

    Args:
        q:     (B, n_heads,    T, head_dim) — query tensor
        k:     (B, n_kv_heads, S, head_dim) — key tensor (expanded internally)
        v:     (B, n_kv_heads, S, head_dim) — value tensor (expanded internally)
        mask:  additive bias broadcastable to (B, n_heads, T, S)
               0.0 = attend, float('-inf') = causal mask / padding
        scale: QK scale; defaults to 1/sqrt(head_dim)

    Returns:
        (B, n_heads, T, head_dim), dtype matches q
    """
    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    head_dim = q.shape[-1]
    n_groups = n_heads // n_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # GQA: expand K, V so every Q head has a dedicated KV head
    k = repeat_kv(k, n_groups)  # (B, n_heads, S, head_dim)
    v = repeat_kv(v, n_groups)

    # ── Step 1: Upcast to FP32 ────────────────────────────────────────────
    # BF16 has 7 mantissa bits; QK^T products lose precision without upcast.
    # This matches the Triton kernel's mandatory FP32 upcast before tl.exp().
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    # ── Step 2: Scaled dot-product scores ─────────────────────────────────
    # Explicit matmul (not F.sdpa) so we control dtype at every step.
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # (B, H, T, S)

    # ── Step 3: Additive attention mask ───────────────────────────────────
    if mask is not None:
        scores = scores + mask.float()

    # ── Step 4: Online softmax in FP32 ────────────────────────────────────
    # Subtract row max before exp() to prevent overflow (Flash Attention style).
    # This is the scalar-complete equivalent of the tile-by-tile running-max
    # accumulation in the Triton kernel. Same numerical result, same invariant:
    # exp() is never called on un-shifted scores.
    row_max = scores.amax(dim=-1, keepdim=True)   # (B, H, T, 1)
    scores = scores - row_max                      # shift to (-inf, 0]
    weights = torch.exp(scores)                    # all-positive, in FP32
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # ── Step 5: Weighted sum over values ──────────────────────────────────
    out_f = torch.matmul(weights, v_f)             # (B, H, T, head_dim) in FP32

    # ── Step 6: Downcast — only after full FP32 normalization ─────────────
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
        GQA forward pass with FP32 online softmax.

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
