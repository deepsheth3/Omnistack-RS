"""
OmniStack-RS — Single Source of Truth: OmniConfig

All hyperparameters live here. Every module imports from this file —
no magic numbers scattered in kernel code.

Invariants validated in __post_init__ (MANIFEST.md §NEVER VIOLATE):
  head_dim    — must be a power of 2 ≥ 64 for WHT butterfly alignment
  quant_bits  — INT4 (4); nibble packing is hardwired, not a free parameter
  n_heads % n_kv_heads == 0 — GQA structural requirement
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OmniConfig:
    """
    Frozen configuration dataclass: all fields are set at construction time.
    Use OmniConfig() for the production defaults or override for testing.
    """

    # ── Invariants (validated below, do not change without versioning) ─────
    head_dim: int = 128       # D_head; 128 = 2^7, required for 7-level WHT butterfly
    quant_bits: int = 4       # INT4 nibble: 2 codes/byte, O(1) branch-free unpack
    qjl_proj_dim: int = 64   # Rademacher sign projections for 1-bit QJL residual
    manifold_rank: int = 32   # Gr(32, 128) captures ≥95% variance in production

    # ── Attention architecture ─────────────────────────────────────────────
    n_heads: int = 32         # total query heads
    n_kv_heads: int = 8       # KV heads (GQA: n_heads must be divisible by n_kv_heads)
    hidden_dim: int = 4096    # D_model

    # ── Shadow LoRA (Stage 2 / 5) ──────────────────────────────────────────
    lora_rank: int = 16       # r: LoRA rank on Q and V projections
    lora_alpha: float = 32.0  # α: scaling; effective scale = α/r = 2.0 at defaults

    # ── KV cache compression (Stage 5) ────────────────────────────────────
    kv_block_size: int = 256  # elements per INT4 quantization block
    page_size: int = 16       # tokens per KV cache page (PagedKVCache)

    # ── Manifold & temporal decay (Stage 1 / 4) ────────────────────────────
    half_life_days: float = 30.0  # exponential decay half-life for viewing history
    ambient_dim: int = 128        # embedding dim before manifold projection

    def __post_init__(self) -> None:
        # head_dim: power of 2 and ≥ 64 (WHT butterfly requires 2^k levels)
        if self.head_dim < 64 or (self.head_dim & (self.head_dim - 1)) != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be a power of 2 and ≥ 64 "
                "(required for Walsh-Hadamard Transform butterfly)"
            )
        # GQA structural requirement
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by "
                f"n_kv_heads={self.n_kv_heads} for GQA"
            )
        # Sanity: quant_bits must be 4 (nibble packing is hardwired)
        if self.quant_bits != 4:
            raise ValueError(
                f"quant_bits must be 4 (INT4 nibble); got {self.quant_bits}. "
                "The kernel packs two 4-bit codes per byte — 3-bit or 5-bit are FORBIDDEN."
            )

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def n_groups(self) -> int:
        """GQA group size: how many Q heads share one KV head."""
        return self.n_heads // self.n_kv_heads

    @property
    def lora_scale(self) -> float:
        """Effective LoRA merge scale: α / r."""
        return self.lora_alpha / self.lora_rank

    @property
    def vram_reduction_manifold(self) -> float:
        """VRAM reduction from manifold pruning: ambient_dim / manifold_rank."""
        return self.ambient_dim / self.manifold_rank

    @property
    def vram_reduction_quant(self) -> float:
        """VRAM reduction from INT4+QJL: 16 bits → (quant_bits + 1) bits."""
        return 16.0 / (self.quant_bits + 1)

    @property
    def vram_reduction_total(self) -> float:
        """Combined VRAM reduction (Stage 4 × Stage 5). Target: 12.8×."""
        return self.vram_reduction_manifold * self.vram_reduction_quant


# Production singleton
DEFAULT_CONFIG = OmniConfig()
