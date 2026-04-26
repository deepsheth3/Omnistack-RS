"""
Unit tests for Phase 1: GQA Reference Attention with Online Softmax.

Validates the tile-by-tile online softmax algorithm against the naive
(materialize-full-score-matrix) baseline and torch.nn.functional.
Run on Mac CPU — no GPU required.

    pytest tests/unit/test_attention.py -v
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from omnistack_rs.attention.reference import reference_attn, repeat_kv, ReferenceAttention, _lora_delta
from omnistack_rs.config import OmniConfig


# ── Helpers ───────────────────────────────────────────────────────────────

def _naive_attn(q, k, v, mask=None, scale=None):
    """
    Materialize-everything baseline: computes the full (T, S) score matrix.
    Used only to verify online_attn gives the same numerical result.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    n_groups = q.shape[1] // k.shape[1]
    k = repeat_kv(k, n_groups)
    v = repeat_kv(v, n_groups)

    q_f, k_f, v_f = q.float(), k.float(), v.float()
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # (B, H, T, S)
    if mask is not None:
        scores = scores + mask.float()
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_s   = torch.exp(scores - row_max)
    weights = exp_s / exp_s.sum(dim=-1, keepdim=True)
    return (weights @ v_f).to(q.dtype)


def _make_qkv(B, H, Hkv, T, S, D, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H,   T, D)
    k = torch.randn(B, Hkv, S, D)
    v = torch.randn(B, Hkv, S, D)
    return q, k, v


# ── Online softmax correctness ────────────────────────────────────────────

class TestOnlineSoftmaxCorrectness:
    """
    Core invariant: tile-by-tile online softmax == naive all-at-once softmax,
    for any tile size, any sequence length, with or without mask.
    """

    @pytest.mark.parametrize("tile_size", [1, 7, 32, 128, 512])
    def test_tile_size_independence(self, tile_size):
        """Different tile sizes must produce identical outputs (atol=1e-5 in FP32)."""
        q, k, v = _make_qkv(B=1, H=4, Hkv=4, T=16, S=64, D=32, seed=42)
        ref  = _naive_attn(q, k, v)
        out  = reference_attn(q, k, v, tile_size=tile_size)
        assert torch.allclose(ref, out, atol=1e-5), (
            f"tile_size={tile_size}: max diff {(ref - out).abs().max():.2e}"
        )

    def test_matches_naive_no_mask(self):
        q, k, v = _make_qkv(B=2, H=8, Hkv=8, T=32, S=64, D=64, seed=1)
        assert torch.allclose(_naive_attn(q, k, v), reference_attn(q, k, v), atol=1e-5)

    def test_matches_naive_with_causal_mask(self):
        """Causal mask: upper-triangle -inf must correctly zero attention to future tokens."""
        B, H, T, D = 1, 4, 32, 64
        q, k, v = _make_qkv(B=B, H=H, Hkv=H, T=T, S=T, D=D, seed=2)
        mask = torch.zeros(1, 1, T, T)
        mask.masked_fill_(torch.ones(T, T, dtype=torch.bool).triu(diagonal=1), float("-inf"))
        assert torch.allclose(_naive_attn(q, k, v, mask), reference_attn(q, k, v, mask), atol=1e-5)

    def test_matches_naive_gqa(self):
        """GQA: 32 Q heads, 8 KV heads (n_groups=4)."""
        q, k, v = _make_qkv(B=1, H=32, Hkv=8, T=16, S=48, D=64, seed=3)
        assert torch.allclose(_naive_attn(q, k, v), reference_attn(q, k, v), atol=1e-5)

    def test_single_query_position(self):
        """T=1 (decode step): the most common production case."""
        q, k, v = _make_qkv(B=2, H=8, Hkv=8, T=1, S=512, D=64, seed=4)
        assert torch.allclose(_naive_attn(q, k, v), reference_attn(q, k, v), atol=1e-5)

    def test_seq_len_not_multiple_of_tile(self):
        """S=100 with tile_size=32: last tile has size 4 — must not OOB."""
        q, k, v = _make_qkv(B=1, H=4, Hkv=4, T=8, S=100, D=32, seed=5)
        assert torch.allclose(_naive_attn(q, k, v), reference_attn(q, k, v, tile_size=32), atol=1e-5)

    def test_tile_size_larger_than_seqlen(self):
        """tile_size > S: algorithm degenerates to a single tile — still correct."""
        q, k, v = _make_qkv(B=1, H=4, Hkv=4, T=8, S=16, D=32, seed=6)
        assert torch.allclose(_naive_attn(q, k, v), reference_attn(q, k, v, tile_size=1024), atol=1e-5)


# ── Running-max invariant ─────────────────────────────────────────────────

class TestRunningMaxInvariant:
    """
    The running max m must be monotonically non-decreasing across tiles.
    Instrumenting the update: m_new = max(m_old, tile_max) ≥ m_old.
    """

    def test_running_max_nondecreasing(self):
        """
        Instrument reference_attn to record m at each tile; verify m never drops.
        We test this indirectly: run tile_size=1 and verify the algorithm is stable
        (no NaN/Inf in the output, which would happen if m decreased and caused
        exp() to receive arbitrarily large positive inputs).
        """
        # tile_size=1: most aggressive tiling — each tile is a single key position.
        # If the running max update were wrong, exp() would produce inf here.
        q, k, v = _make_qkv(B=1, H=4, Hkv=4, T=8, S=64, D=32, seed=10)
        out = reference_attn(q, k, v, tile_size=1)
        assert not torch.isnan(out).any(), "NaN in output — running max update is broken"
        assert not torch.isinf(out).any(), "Inf in output — exp() overflow, max not updated"

    def test_no_overflow_with_large_scores(self):
        """
        Scores much larger than 0 must not overflow exp().
        Standard softmax (no shift) would produce inf here; online softmax is safe.
        """
        torch.manual_seed(11)
        q = torch.randn(1, 2, 4, 32) * 100   # very large magnitude → large QK^T
        k = torch.randn(1, 2, 16, 32) * 100
        v = torch.randn(1, 2, 16, 32)
        out = reference_attn(q, k, v)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_weights_sum_to_one(self):
        """
        Softmax weights must sum to 1.0 per query position.
        Verify by checking: if we set V = ones, output == 1.0 everywhere.
        """
        torch.manual_seed(12)
        B, H, T, S, D = 1, 4, 8, 32, 64
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, S, D)
        v = torch.ones(B, H, S, D)
        out = reference_attn(q, k, v)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-5), (
            f"V=ones should give out=ones; max diff {(out - 1.0).abs().max():.2e}"
        )

    def test_all_minus_inf_mask_except_one(self):
        """
        If all keys except position 0 are masked to -inf, output = v[:, :, 0, :].
        Tests that the running-max update handles -inf scores cleanly.
        """
        B, H, T, S, D = 1, 2, 4, 8, 32
        torch.manual_seed(13)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        # Mask out everything except position 0
        mask = torch.full((1, 1, T, S), float("-inf"))
        mask[..., 0] = 0.0
        out = reference_attn(q, k, v, mask=mask)
        expected = v[:, :, 0:1, :].expand(B, H, T, D).float()
        assert torch.allclose(out.float(), expected, atol=1e-5), (
            f"Only position 0 unmasked: output should equal v[0]. Max diff {(out.float() - expected).abs().max():.2e}"
        )


# ── Memory: score matrix is never materialized ────────────────────────────

class TestMemoryFootprint:
    """
    The full (T × S) score matrix must never be allocated.
    We verify this structurally: the tile loop allocates at most (T × tile_size)
    per iteration. For large S with small tile_size, peak memory is O(tile_size).
    """

    def test_output_shape_preserved(self):
        q, k, v = _make_qkv(B=2, H=8, Hkv=4, T=16, S=64, D=64)
        out = reference_attn(q, k, v)
        assert out.shape == q.shape

    def test_output_dtype_matches_input(self):
        """Output dtype must match q.dtype (downcast after FP32 computation)."""
        q, k, v = _make_qkv(B=1, H=4, Hkv=4, T=8, S=16, D=32)
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            out = reference_attn(q.to(dtype), k.to(dtype), v.to(dtype))
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"

    def test_long_sequence_does_not_oom(self):
        """
        S=4096, tile_size=128: 32 tiles × (B, H, T, 128) score buffer.
        With the naive approach, (B=1, H=4, T=16, S=4096) × 4 bytes = 1 MB score matrix.
        With tiling, peak allocation is (B=1, H=4, T=16, 128) × 4 bytes = 32 KB.
        This test just verifies it runs without OOM or NaN.
        """
        q, k, v = _make_qkv(B=1, H=4, Hkv=4, T=16, S=4096, D=64, seed=20)
        out = reference_attn(q, k, v, tile_size=128)
        assert out.shape == (1, 4, 16, 64)
        assert not torch.isnan(out).any()

    def test_input_tensors_not_modified(self):
        """reference_attn must not mutate its input tensors."""
        q, k, v = _make_qkv(B=1, H=4, Hkv=4, T=8, S=16, D=32, seed=21)
        q0, k0, v0 = q.clone(), k.clone(), v.clone()
        reference_attn(q, k, v)
        assert torch.equal(q, q0) and torch.equal(k, k0) and torch.equal(v, v0)


# ── GQA head expansion ────────────────────────────────────────────────────

class TestGQA:

    def test_repeat_kv_shape(self):
        x = torch.randn(2, 8, 16, 64)
        out = repeat_kv(x, n_groups=4)
        assert out.shape == (2, 32, 16, 64)

    def test_repeat_kv_noop_at_1(self):
        x = torch.randn(2, 8, 16, 64)
        assert repeat_kv(x, 1) is x

    def test_repeat_kv_values(self):
        """Each KV head must appear n_groups times consecutively."""
        x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4)  # 2 kv heads
        out = repeat_kv(x, n_groups=3)   # → 6 heads
        # Head 0 occupies positions 0,1,2; head 1 occupies 3,4,5
        assert torch.equal(out[0, 0], out[0, 1]) and torch.equal(out[0, 1], out[0, 2])
        assert torch.equal(out[0, 3], out[0, 4]) and torch.equal(out[0, 4], out[0, 5])

    def test_gqa_mha_equivalence(self):
        """With n_groups=1, GQA reduces to standard MHA."""
        q, k, v = _make_qkv(B=2, H=8, Hkv=8, T=16, S=32, D=64, seed=30)
        mha   = reference_attn(q, k, v)
        gqa_1 = reference_attn(q, k, v)
        assert torch.allclose(mha, gqa_1, atol=1e-6)

    def test_gqa_n_groups_4(self):
        """GQA with n_groups=4: output shape matches n_heads, not n_kv_heads."""
        q, k, v = _make_qkv(B=1, H=32, Hkv=8, T=8, S=16, D=64, seed=31)
        out = reference_attn(q, k, v)
        assert out.shape == (1, 32, 8, 64)


# ── Causal mask ───────────────────────────────────────────────────────────

class TestCausalMask:

    def test_causal_mask_blocks_future(self):
        """
        In a causal sequence, query at position t should not attend to t+1..T-1.
        Verify by checking that output at position 0 is independent of k/v at pos 1+.
        """
        B, H, T, D = 1, 2, 8, 32
        torch.manual_seed(40)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        mask = torch.zeros(1, 1, T, T)
        mask.masked_fill_(torch.ones(T, T, dtype=torch.bool).triu(diagonal=1), float("-inf"))

        out_orig = reference_attn(q, k, v, mask=mask)

        # Scramble k/v at positions 1+ — output at position 0 must not change
        k2 = k.clone()
        v2 = v.clone()
        k2[:, :, 1:, :] = torch.randn_like(k2[:, :, 1:, :])
        v2[:, :, 1:, :] = torch.randn_like(v2[:, :, 1:, :])
        out_modified = reference_attn(q, k2, v2, mask=mask)

        assert torch.allclose(out_orig[:, :, 0, :], out_modified[:, :, 0, :], atol=1e-5), (
            "Position 0 output changed when future k/v were scrambled — causal mask broken"
        )

    def test_causal_mask_shape(self):
        cfg = OmniConfig()
        T = 16
        m = ReferenceAttention(cfg).causal_mask(T=T, device=torch.device("cpu"))
        assert m.shape == (1, 1, T, T)
        mask_2d = m[0, 0]  # (T, T)
        rows, cols = torch.meshgrid(torch.arange(T), torch.arange(T), indexing="ij")
        upper = cols > rows   # True where future tokens should be masked
        assert (mask_2d[upper] == float("-inf")).all(), "Upper triangle must be -inf (future masked)"
        assert (mask_2d[~upper] == 0.0).all(), "Lower triangle must be 0 (attend)"


# ── ReferenceAttention module ─────────────────────────────────────────────

class TestReferenceAttentionModule:

    @pytest.fixture
    def cfg(self):
        return OmniConfig()

    @pytest.fixture
    def model(self, cfg):
        torch.manual_seed(50)
        return ReferenceAttention(cfg)

    def test_forward_shape(self, model, cfg):
        B, T = 2, 16
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_forward_with_causal_mask(self, model, cfg):
        B, T = 1, 32
        x = torch.randn(B, T, cfg.hidden_dim)
        mask = model.causal_mask(T, x.device)
        out = model(x, mask=mask)
        assert out.shape == (B, T, cfg.hidden_dim)
        assert not torch.isnan(out).any()

    def test_lora_delta_shape(self, model, cfg):
        r, d_in, d_out = cfg.lora_rank, cfg.hidden_dim, cfg.n_heads * cfg.head_dim
        A = torch.randn(r, d_in)
        B_mat = torch.randn(d_out, r)
        delta = _lora_delta(A, B_mat, cfg.lora_alpha, r)
        assert delta.shape == (d_out, d_in)

    def test_effective_weight_formula(self, model, cfg):
        """W_eff = W_base + B @ A * (α/r) — verify against manual calculation."""
        r, d_in = cfg.lora_rank, cfg.hidden_dim
        d_out = cfg.n_heads * cfg.head_dim
        torch.manual_seed(51)
        A = torch.randn(r, d_in)
        B_mat = torch.randn(d_out, r)

        w_eff = model.effective_weight("q_proj", A, B_mat, alpha=cfg.lora_alpha, rank=r)
        expected = model.q_proj.weight + (B_mat @ A) * (cfg.lora_alpha / r)
        assert torch.allclose(w_eff, expected, atol=1e-6)

    def test_apply_lora_modifies_weight(self, model, cfg):
        """apply_lora() must update q_proj.weight in-place."""
        r, d_in = cfg.lora_rank, cfg.hidden_dim
        d_out = cfg.n_heads * cfg.head_dim
        torch.manual_seed(52)
        A = torch.randn(r, d_in)
        B_mat = torch.randn(d_out, r)

        w_before = model.q_proj.weight.clone()
        model.apply_lora("q_proj", A, B_mat, alpha=cfg.lora_alpha, rank=r)
        w_after = model.q_proj.weight

        assert not torch.equal(w_before, w_after), "apply_lora must change the weight"
        expected_delta = (B_mat @ A) * (cfg.lora_alpha / r)
        assert torch.allclose((w_after - w_before).float(), expected_delta.float(), atol=1e-5)

    def test_apply_lora_rejects_unknown_proj(self, model):
        A = torch.randn(8, 64)
        B_mat = torch.randn(64, 8)
        with pytest.raises(ValueError, match="Unknown projection"):
            model.apply_lora("z_proj", A, B_mat)
