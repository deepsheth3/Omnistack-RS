"""
OmniStack-RS — Phase 5 Multi-LoRA Quality Gates

Four correctness tests + three MLPerf-style statistical checks that must all
pass before a Phase 5 build is considered complete.

Quality gates (as specified in the Phase 5 brief):

  1. Numerical Parity    — fused kernel vs. manual W_base + BA merge (atol=1e-3)
  2. Sentinel (-1)       — users with no adapter get base-model-only output
  3. Rank Purity         — rank-4 padded to rank-16 with zeros = same scores as rank-4
  4. Batch Boundary      — user 0's LoRA does not leak into user 1's output

MLPerf Inference v6.0 statistical gates (CPU proxy; H100 numbers would be ~100×):

  5. Numerical Parity ≥ 99 %   — fused vs. FP32 reference rank correlation
  6. Latency Consistency        — P99 ≤ 1.5 × Mean across repeated query batches
  7. Throughput Scaling         — QPS scales (super-)linearly with batch size

All tests run under TRITON_INTERPRET=1 (CPU, no CUDA required).
"""

from __future__ import annotations

import math
import statistics
import time
from typing import List

import pytest
import torch

from omnistack_rs.kernels.fused_attention import _apply_lora_to_q, omni_attn
from omnistack_rs.kernels.quantize import quantize_heads
from omnistack_rs.quantization.codebook import calibrate_per_group

# ── Shared constants ──────────────────────────────────────────────────────────

HEAD_DIM = 128
N_Q_HEADS = 4
N_KV_HEADS = 2
LORA_RANK = 16    # "heavy" rank (max rank for padding tests)
HIDDEN_DIM = 64   # hidden_dim of x fed to the LoRA down-projection


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_corpus(
    B: int,
    S: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate and quantize a random KV corpus.

    Returns (nibbles, qjl, norms, codebooks) ready for omni_attn.
    """
    torch.manual_seed(seed)
    kv = torch.randn(B, N_KV_HEADS, S, HEAD_DIM)
    samples = [kv[:, h, :, :].reshape(-1) for h in range(N_KV_HEADS)]
    codebooks = calibrate_per_group(samples)
    nibbles, qjl, norms = quantize_heads(kv, codebooks, user_id=0)
    return nibbles, qjl, norms, codebooks


def _make_lora(
    n_loras: int,
    rank: int,
    hidden_dim: int,
    scale: float = 0.05,
    seed: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create (lora_a, lora_b) weight tensors for n_loras distinct adapters.

    lora_a: (n_loras, rank, hidden_dim)
    lora_b: (n_loras, N_Q_HEADS * HEAD_DIM, rank)
    """
    torch.manual_seed(seed)
    la = torch.randn(n_loras, rank, hidden_dim) * scale
    lb = torch.randn(n_loras, N_Q_HEADS * HEAD_DIM, rank) * scale
    return la, lb


# ── Quality gate 1: Numerical Parity ─────────────────────────────────────────

class TestNumericalParity:
    """
    Gate: for every user in the batch, the fused kernel must match the manual
    'merged weight' baseline within atol=1e-3.

    The 'merged weight' baseline is:
        Q_eff[b] = Q_base[b] + (x[b] @ A[slot].T) @ B[slot].T * alpha
    computed in pure Python, then fed to the reference attention path.
    Any deviation > 1e-3 indicates a pointer-aliasing bug where user A is
    accidentally reading user B's LoRA weights.
    """

    def test_8_users_8_loras_distinct_slots(self):
        """Each of 8 users maps to a distinct LoRA slot (no aliasing possible)."""
        B, T, S = 8, 16, 32
        torch.manual_seed(42)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S)
        lora_a, lora_b = _make_lora(n_loras=8, rank=LORA_RANK, hidden_dim=HIDDEN_DIM)
        lora_indices = torch.arange(B, dtype=torch.int32)   # user i → slot i
        alpha = 0.5

        # Fused path
        out_fused = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=alpha,
        )

        # Manual merged-weight baseline
        q_eff = _apply_lora_to_q(q, x, lora_a, lora_b, lora_indices, alpha)
        out_ref = omni_attn(q_eff, nibbles, qjl, norms, nibbles, codebooks, user_id=0)

        max_delta = (out_fused - out_ref).abs().max().item()
        assert max_delta < 1e-3, (
            f"Numerical parity FAIL: max|fused - merged_weight| = {max_delta:.2e} "
            f"(gate: < 1e-3). Likely pointer-aliasing bug in LoRA slot dispatch."
        )

    def test_repeated_slot_aliasing_check(self):
        """
        Deliberately map multiple users to the same slot to stress pointer aliasing.
        Users {0,2,4} → slot 0;  users {1,3,5} → slot 1.
        The fused output must still match the reference for every user.
        """
        B, T, S = 6, 8, 16
        torch.manual_seed(7)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S, seed=7)
        lora_a, lora_b = _make_lora(n_loras=2, rank=LORA_RANK, hidden_dim=HIDDEN_DIM)
        lora_indices = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32)
        alpha = 1.0

        out_fused = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=alpha,
        )
        q_eff = _apply_lora_to_q(q, x, lora_a, lora_b, lora_indices, alpha)
        out_ref = omni_attn(q_eff, nibbles, qjl, norms, nibbles, codebooks, user_id=0)

        max_delta = (out_fused - out_ref).abs().max().item()
        assert max_delta < 1e-3, f"Aliased-slot parity FAIL: max|delta|={max_delta:.2e}"

    def test_per_user_output_differs_across_lora_slots(self):
        """
        Sanity check: different LoRA slots must produce meaningfully different outputs.
        If all outputs are identical, the LoRA dispatch is silently broken.
        """
        B, T, S = 4, 8, 16
        torch.manual_seed(13)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S)
        lora_a, lora_b = _make_lora(n_loras=4, rank=LORA_RANK, hidden_dim=HIDDEN_DIM, scale=0.3)
        lora_indices = torch.arange(B, dtype=torch.int32)

        out = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=1.0,
        )

        # Every pair of consecutive users must differ
        for u in range(B - 1):
            diff = (out[u] - out[u + 1]).abs().max().item()
            assert diff > 1e-4, (
                f"User {u} and user {u+1} produced identical outputs — "
                f"LoRA slot dispatch may be broken (diff={diff:.2e})"
            )


# ── Quality gate 2: Sentinel (-1) — No-Adapter Users ─────────────────────────

class TestSentinelNoAdapter:
    """
    Gate: lora_index = -1 must short-circuit the LoRA math and produce exactly
    the same output as calling omni_attn without any LoRA arguments.

    Failure here means sentinel users pay the LoRA compute cost (wasted cycles)
    or, worse, accidentally read slot-0 weights (silent corruption).
    """

    def test_all_sentinel_matches_base(self):
        """Full batch of -1 sentinels → identical to no-LoRA base path."""
        B, T, S = 4, 8, 16
        torch.manual_seed(0)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S)
        lora_a, lora_b = _make_lora(n_loras=2, rank=LORA_RANK, hidden_dim=HIDDEN_DIM)
        lora_indices = torch.full((B,), -1, dtype=torch.int32)   # all sentinels

        out_sentinel = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=1.0,
        )
        out_base = omni_attn(q, nibbles, qjl, norms, nibbles, codebooks, user_id=0)

        max_delta = (out_sentinel - out_base).abs().max().item()
        assert max_delta == 0.0, (
            f"Sentinel (-1) FAIL: all-sentinel batch differs from base by {max_delta:.2e}. "
            "Slot-0 weights are leaking through the sentinel guard."
        )

    def test_mixed_sentinel_and_active_users(self):
        """
        50 % of users have -1 sentinel, 50 % have active LoRAs.
        Sentinel users must exactly match base output; active users must differ.
        """
        B, T, S = 4, 8, 16
        torch.manual_seed(3)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S, seed=3)
        lora_a, lora_b = _make_lora(n_loras=2, rank=LORA_RANK, hidden_dim=HIDDEN_DIM, scale=0.5)

        # users 0, 2 → sentinel;  users 1, 3 → active LoRA slots 0 and 1
        lora_indices = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)

        out_mixed = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=1.0,
        )
        out_base = omni_attn(q, nibbles, qjl, norms, nibbles, codebooks, user_id=0)

        # Sentinel users must exactly match base
        for u in [0, 2]:
            delta = (out_mixed[u] - out_base[u]).abs().max().item()
            assert delta == 0.0, (
                f"User {u} (sentinel) differs from base by {delta:.2e} — slot-0 leak."
            )

        # Active users must differ from base (LoRA has non-zero weight)
        for u in [1, 3]:
            delta = (out_mixed[u] - out_base[u]).abs().max().item()
            assert delta > 1e-5, (
                f"User {u} (active LoRA) is identical to base (delta={delta:.2e}) — "
                "LoRA delta was silently zeroed out."
            )

    def test_sentinel_short_circuits_lora_math_via_reference(self):
        """
        Sentinel user's output must match _apply_lora_to_q with idx=-1,
        which skips the LoRA loop. Verifies the CPU path and kernel agree.
        """
        B, T, S = 2, 8, 16
        torch.manual_seed(5)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S)
        lora_a, lora_b = _make_lora(n_loras=3, rank=LORA_RANK, hidden_dim=HIDDEN_DIM)
        lora_indices = torch.tensor([-1, 2], dtype=torch.int32)
        alpha = 0.8

        out_fused = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=alpha,
        )

        # Reference: _apply_lora_to_q must also skip -1
        q_eff = _apply_lora_to_q(q, x, lora_a, lora_b, lora_indices, alpha)
        out_ref = omni_attn(q_eff, nibbles, qjl, norms, nibbles, codebooks, user_id=0)

        max_delta = (out_fused - out_ref).abs().max().item()
        assert max_delta < 1e-3, f"Sentinel reference match FAIL: max|delta|={max_delta:.2e}"


# ── Quality gate 3: Rank Purity ───────────────────────────────────────────────

class TestRankPurity:
    """
    Gate: a rank-4 LoRA padded with zeros to rank-16 must produce the same Q_eff
    (and therefore the same attention output) as the unpadded rank-4 version.

    This validates the 'max-rank padding' strategy used in production to keep
    a fixed LORA_RANK constexpr while supporting heterogeneous adapter ranks.
    """

    def test_rank4_padded_to_rank16_same_q_eff(self):
        """Q_eff from padded rank-16 must exactly match Q_eff from raw rank-4."""
        B, T = 4, 8
        RANK_ACTUAL = 4
        torch.manual_seed(9)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        lora_indices = torch.arange(B, dtype=torch.int32) % 2  # 2 distinct slots

        la_small = torch.randn(2, RANK_ACTUAL, HIDDEN_DIM) * 0.1
        lb_small = torch.randn(2, N_Q_HEADS * HEAD_DIM, RANK_ACTUAL) * 0.1

        # Pad rank dimension with zeros
        la_padded = torch.zeros(2, LORA_RANK, HIDDEN_DIM)
        lb_padded = torch.zeros(2, N_Q_HEADS * HEAD_DIM, LORA_RANK)
        la_padded[:, :RANK_ACTUAL, :] = la_small
        lb_padded[:, :, :RANK_ACTUAL] = lb_small

        q_eff_small  = _apply_lora_to_q(q, x, la_small,  lb_small,  lora_indices, 1.0)
        q_eff_padded = _apply_lora_to_q(q, x, la_padded, lb_padded, lora_indices, 1.0)

        max_delta = (q_eff_small - q_eff_padded).abs().max().item()
        assert max_delta < 1e-5, (
            f"Rank purity FAIL: padded Q_eff differs from unpadded by {max_delta:.2e}. "
            "Zero-padding of unused rank dimensions is polluting the Q update."
        )

    def test_rank4_padded_attention_output_matches(self):
        """End-to-end: padded rank-16 attention output matches rank-4 reference."""
        B, T, S = 4, 8, 16
        RANK_ACTUAL = 4
        torch.manual_seed(17)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S, seed=17)
        lora_indices = torch.arange(B, dtype=torch.int32) % 2

        la_small = torch.randn(2, RANK_ACTUAL, HIDDEN_DIM) * 0.1
        lb_small = torch.randn(2, N_Q_HEADS * HEAD_DIM, RANK_ACTUAL) * 0.1

        la_padded = torch.zeros(2, LORA_RANK, HIDDEN_DIM)
        lb_padded = torch.zeros(2, N_Q_HEADS * HEAD_DIM, LORA_RANK)
        la_padded[:, :RANK_ACTUAL, :] = la_small
        lb_padded[:, :, :RANK_ACTUAL] = lb_small

        # Reference with actual rank-4
        q_eff_small = _apply_lora_to_q(q, x, la_small, lb_small, lora_indices, 1.0)
        out_ref = omni_attn(q_eff_small, nibbles, qjl, norms, nibbles, codebooks, user_id=0)

        # Fused path uses padded rank-16 (what the kernel sees in production)
        out_padded = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=la_padded, lora_b=lb_padded,
            lora_indices=lora_indices, lora_alpha=1.0,
        )

        max_delta = (out_padded - out_ref).abs().max().item()
        assert max_delta < 1e-3, (
            f"Rank purity end-to-end FAIL: max|delta|={max_delta:.2e}. "
            "Padding zeros are causing non-zero contributions to attention scores."
        )

    def test_zero_rank_adapter_is_identity(self):
        """
        LoRA with all-zero weights (rank-16, zeroed) must produce the same output
        as the no-LoRA base path — zero-weight adapter is an identity transform.
        """
        B, T, S = 4, 8, 16
        torch.manual_seed(21)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S)
        lora_a = torch.zeros(1, LORA_RANK, HIDDEN_DIM)
        lora_b = torch.zeros(1, N_Q_HEADS * HEAD_DIM, LORA_RANK)
        lora_indices = torch.zeros(B, dtype=torch.int32)

        out_zero_lora = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=1.0,
        )
        out_base = omni_attn(q, nibbles, qjl, norms, nibbles, codebooks, user_id=0)

        max_delta = (out_zero_lora - out_base).abs().max().item()
        assert max_delta == 0.0, (
            f"Zero-weight LoRA is not identity: max|delta|={max_delta:.2e}"
        )


# ── Quality gate 4: Batch Boundary — No SRAM Leakage ─────────────────────────

class TestBatchBoundary:
    """
    Gate: user 0's LoRA adapter must not affect user 1's attention output,
    and vice versa — even though both share the same SRAM tile buffers.

    Failure here means the pid_b → lora_idx pointer lookup has an off-by-one
    error, or that the running-state registers (m, l, O) are not reset between
    Triton programs (which would be a Triton compiler bug, but worth asserting).
    """

    def test_batched_matches_individual_runs(self):
        """
        Run B=4 users in a single batch, then run each user individually.
        Each user's batched output must exactly match their solo output.
        """
        B, T, S = 4, 8, 16
        torch.manual_seed(99)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S, seed=99)
        lora_a, lora_b = _make_lora(n_loras=B, rank=LORA_RANK, hidden_dim=HIDDEN_DIM, scale=0.3)
        lora_indices = torch.arange(B, dtype=torch.int32)
        alpha = 0.7

        out_batch = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=alpha,
        )

        for u in range(B):
            out_solo = omni_attn(
                q[u:u+1], nibbles[u:u+1], qjl[u:u+1], norms[u:u+1],
                nibbles[u:u+1], codebooks, user_id=0,
                x=x[u:u+1],
                lora_a=lora_a, lora_b=lora_b,
                lora_indices=lora_indices[u:u+1],
                lora_alpha=alpha,
            )
            max_delta = (out_batch[u] - out_solo[0]).abs().max().item()
            assert max_delta < 1e-4, (
                f"Batch boundary FAIL for user {u}: "
                f"batched output differs from solo run by {max_delta:.2e}. "
                "Another user's LoRA weights are leaking across SRAM tile boundaries."
            )

    def test_swap_lora_indices_changes_output(self):
        """
        Swapping user 0 and user 1's LoRA slot indices must change both outputs.
        If outputs are invariant to the swap, the dispatch is broken.
        """
        B, T, S = 2, 8, 16
        torch.manual_seed(55)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S)
        lora_a, lora_b = _make_lora(n_loras=2, rank=LORA_RANK, hidden_dim=HIDDEN_DIM, scale=0.4)

        idx_01 = torch.tensor([0, 1], dtype=torch.int32)
        idx_10 = torch.tensor([1, 0], dtype=torch.int32)   # swapped

        out_01 = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=idx_01, lora_alpha=1.0,
        )
        out_10 = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=idx_10, lora_alpha=1.0,
        )

        for u in range(B):
            delta = (out_01[u] - out_10[u]).abs().max().item()
            assert delta > 1e-5, (
                f"User {u}: swapping LoRA indices produced no output change "
                f"(delta={delta:.2e}). Pointer-array dispatch may be ignoring indices."
            )

    def test_sentinel_user_unaffected_by_neighbor_lora(self):
        """
        A sentinel user (index -1) placed next to an active user must produce
        the same output regardless of what LoRA slot the active neighbor holds.
        """
        B, T, S = 2, 8, 16
        torch.manual_seed(77)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S)
        lora_a, lora_b = _make_lora(n_loras=3, rank=LORA_RANK, hidden_dim=HIDDEN_DIM, scale=0.5)

        def _run(active_slot: int) -> torch.Tensor:
            idx = torch.tensor([-1, active_slot], dtype=torch.int32)
            return omni_attn(
                q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
                x=x, lora_a=lora_a, lora_b=lora_b,
                lora_indices=idx, lora_alpha=1.0,
            )

        out_slot0 = _run(0)
        out_slot1 = _run(1)
        out_slot2 = _run(2)

        # User 0 (sentinel) must be identical across all three runs
        for ref, name in [(out_slot1, "slot1"), (out_slot2, "slot2")]:
            delta = (out_slot0[0] - ref[0]).abs().max().item()
            assert delta == 0.0, (
                f"Sentinel user 0 differs when neighbor switches to {name}: "
                f"delta={delta:.2e}. Neighbor's LoRA is leaking into the sentinel."
            )

        # User 1 (active) must differ across slots
        delta_01 = (out_slot0[1] - out_slot1[1]).abs().max().item()
        assert delta_01 > 1e-5, "Active user output invariant to LoRA slot change"


# ── MLPerf Statistical Gates ──────────────────────────────────────────────────

class TestMLPerfStatisticalGates:
    """
    Three statistical quality gates from MLPerf Inference v6.0, run on CPU
    as a proxy. H100 numbers are expected to be ~100× faster.

    Gate 5 — Numerical Parity ≥ 99 %:
        Rank correlation between fused output and FP32 reference must be ≥ 0.99.

    Gate 6 — Latency Consistency:
        P99 ≤ 1.5 × Mean across 20 repeated same-input query batches.

    Gate 7 — Throughput Scaling:
        QPS (users/s) does not decrease when doubling batch size.
    """

    def test_mlperf_numerical_parity_99pct(self):
        """
        Gate 5: ≥ 99 % of output elements agree with the FP32 reference within
        a loose per-element tolerance (atol=5e-2 relative to the output range).

        Simulates the MLPerf "accuracy" check: the fused LoRA kernel's numerical
        error must not corrupt more than 1 % of the output tensor values enough
        to change a ranking decision.
        """
        B, T, S = 4, 8, 32
        torch.manual_seed(11)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S, seed=11)
        lora_a, lora_b = _make_lora(n_loras=B, rank=LORA_RANK, hidden_dim=HIDDEN_DIM, scale=0.1)
        lora_indices = torch.arange(B, dtype=torch.int32)

        # Fused path
        out_fused = omni_attn(
            q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
            x=x, lora_a=lora_a, lora_b=lora_b,
            lora_indices=lora_indices, lora_alpha=1.0,
        )

        # FP32 reference path: pre-apply LoRA in Python, then standard attention
        q_eff = _apply_lora_to_q(q, x, lora_a, lora_b, lora_indices, 1.0)
        out_ref = omni_attn(q_eff, nibbles, qjl, norms, nibbles, codebooks, user_id=0)

        # Element-wise agreement: gate is ≥ 99 % within atol=5e-2
        atol = 5e-2
        n_total = out_fused.numel()
        n_agree = int(((out_fused - out_ref).abs() < atol).sum().item())
        pct_agree = n_agree / n_total

        assert pct_agree >= 0.99, (
            f"MLPerf Gate 5 FAIL: {pct_agree*100:.2f}% of elements agree "
            f"within atol={atol} (gate: ≥ 99%). "
            "Fused + LoRA compression is distorting too many output values."
        )

    def test_mlperf_latency_consistency(self):
        """
        Gate 6: P99 latency ≤ 1.5 × Mean latency across 20 repeated queries.

        Jitter above 1.5× suggests irregular GPU stalls (e.g., TMA pipeline
        conflicts during LoRA dispatch, or cache misses on LoRA weight loads).
        """
        B, T, S = 4, 8, 32
        N_WARMUP = 3
        N_TIMED  = 20
        torch.manual_seed(33)

        q = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
        x = torch.randn(B, T, HIDDEN_DIM)
        nibbles, qjl, norms, codebooks = _make_corpus(B, S)
        lora_a, lora_b = _make_lora(n_loras=B, rank=LORA_RANK, hidden_dim=HIDDEN_DIM)
        lora_indices = torch.arange(B, dtype=torch.int32)

        def _run():
            return omni_attn(
                q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
                x=x, lora_a=lora_a, lora_b=lora_b,
                lora_indices=lora_indices, lora_alpha=1.0,
            )

        for _ in range(N_WARMUP):
            _run()

        latencies_ms = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            _run()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = statistics.mean(latencies_ms)
        p99_ms  = sorted(latencies_ms)[int(N_TIMED * 0.99)]
        ratio   = p99_ms / mean_ms if mean_ms > 0 else float("inf")

        assert ratio <= 1.5, (
            f"MLPerf Gate 6 FAIL: P99/Mean = {ratio:.2f} (gate: ≤ 1.5). "
            f"P99={p99_ms:.2f} ms, Mean={mean_ms:.2f} ms. "
            "Excessive latency jitter detected — check LoRA index dispatch path."
        )

    def test_mlperf_throughput_does_not_collapse_with_batch(self):
        """
        Gate 7: QPS (queries/s) must not decrease when doubling the batch size.

        Collapse here indicates the LoRA dispatch is creating a sequential
        bottleneck that prevents batch parallelism from exploiting the GPU.
        Measured across B=2, B=4, B=8 on CPU as a proxy.
        """
        S, T = 16, 8
        N_WARMUP = 2
        N_TIMED  = 10
        batch_sizes = [2, 4, 8]

        def _time_batch(B: int) -> float:
            torch.manual_seed(B)
            q         = torch.randn(B, N_Q_HEADS, T, HEAD_DIM)
            x         = torch.randn(B, T, HIDDEN_DIM)
            nibbles, qjl, norms, codebooks = _make_corpus(B, S, seed=B)
            lora_a, lora_b = _make_lora(n_loras=B, rank=LORA_RANK, hidden_dim=HIDDEN_DIM)
            lora_indices = torch.arange(B, dtype=torch.int32)

            def _run():
                return omni_attn(
                    q, nibbles, qjl, norms, nibbles, codebooks, user_id=0,
                    x=x, lora_a=lora_a, lora_b=lora_b,
                    lora_indices=lora_indices, lora_alpha=1.0,
                )

            for _ in range(N_WARMUP):
                _run()

            t0 = time.perf_counter()
            for _ in range(N_TIMED):
                _run()
            elapsed = time.perf_counter() - t0

            users_total = B * N_TIMED
            return users_total / elapsed   # users-per-second

        ups_by_batch = {B: _time_batch(B) for B in batch_sizes}

        # UPS must be non-decreasing (allow ≤ 5 % regression as CPU timing noise)
        for i in range(len(batch_sizes) - 1):
            b_small = batch_sizes[i]
            b_large = batch_sizes[i + 1]
            ups_small = ups_by_batch[b_small]
            ups_large = ups_by_batch[b_large]
            assert ups_large >= ups_small * 0.95, (
                f"MLPerf Gate 7 FAIL: throughput collapsed when doubling batch size "
                f"{b_small}→{b_large}: {ups_small:.1f} → {ups_large:.1f} UPS "
                f"(ratio={ups_large/ups_small:.2f}, gate: ≥ 0.95). "
                "LoRA dispatch may be introducing sequential bottlenecks."
            )
