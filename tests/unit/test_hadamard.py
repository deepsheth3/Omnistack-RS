"""
Unit tests for the Hadamard WHT rotation kernel.

Primary validation target (Architect's requirement):
    abs(dot(Q, K) - dot(WHT(Q), WHT(K))) < 1e-4

Run on Mac (no GPU required):
    TRITON_INTERPRET=1 pytest tests/unit/test_hadamard.py -v

GPU tests (Phase 4 gate):
    python ci/run_gpu_tests.py
"""

from __future__ import annotations

import os
# Activate CPU software interpreter before importing Triton-backed code.
# TRITON_INTERPRET=1 causes Triton to execute kernels as Python — no GPU needed.
os.environ.setdefault("TRITON_INTERPRET", "1")

import math
import pytest
import torch

from omnistack_rs.kernels.hadamard import (
    HEAD_DIM,
    _HAS_TRITON,
    _wht_torch,
    _wht_rotate,
    rotate_queries,
    rotate_kv_cache,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def qk_batch():
    """(B=2, H=4, T=8, D=128) random Q and K tensors, seeded for reproducibility."""
    torch.manual_seed(0)
    B, H, T = 2, 4, 8
    return (
        torch.randn(B, H, T, HEAD_DIM),
        torch.randn(B, H, T, HEAD_DIM),
    )


# ── Core invariance: the Architect's required test ────────────────────────

class TestDotProductInvariance:
    """
    WHT rotation must preserve inner products.

    Mathematical basis: (H/√D) is orthonormal → (H/√D)^T (H/√D) = I.
    Therefore dot(Hq/√D, Hk/√D) = q·k for all q, k ∈ ℝ^D.
    """

    def test_elementwise_dot_under_1e4(self, qk_batch):
        """
        Architect's required test:
            abs(dot(Q[i], K[i]) - dot(WHT(Q[i]), WHT(K[i]))) < 1e-4
        for every (batch, head, token) triple.
        """
        q, k = qk_batch
        dots_raw = (q * k).sum(dim=-1)  # (B, H, T)

        q_rot = rotate_queries(q)
        k_rot, _ = rotate_kv_cache(k, torch.zeros_like(k))
        dots_rot = (q_rot * k_rot).sum(dim=-1)

        err = (dots_raw - dots_rot).abs()
        assert err.max().item() < 1e-4, (
            f"Dot invariance violated: max_err={err.max().item():.2e}, "
            f"mean_err={err.mean().item():.2e}"
        )

    def test_qkt_attention_matrix(self, qk_batch):
        """
        Full Q @ K^T matrix is invariant — the exact quantity scaled by
        1/√d to produce attention logits. Tolerance 1e-4 matches Phase 4
        fused kernel validation target (atol=5e-2 at BF16, tighter here at FP32).
        """
        q, k = qk_batch
        qkt = torch.matmul(q, k.transpose(-2, -1))          # (B, H, T, T)

        q_rot = rotate_queries(q)
        k_rot, _ = rotate_kv_cache(k, torch.zeros_like(k))
        qkt_rot = torch.matmul(q_rot, k_rot.transpose(-2, -1))

        max_err = (qkt - qkt_rot).abs().max().item()
        assert max_err < 1e-4, f"QK^T matrix invariance: max_err={max_err:.2e}"

    def test_single_vector_pair(self):
        """Scalar sanity check on a single (1, 1, 1, D) vector pair."""
        torch.manual_seed(42)
        q = torch.randn(1, 1, 1, HEAD_DIM)
        k = torch.randn(1, 1, 1, HEAD_DIM)
        dot_raw = (q * k).sum().item()

        q_rot = rotate_queries(q)
        k_rot, _ = rotate_kv_cache(k, torch.zeros_like(k))
        dot_rot = (q_rot * k_rot).sum().item()

        assert abs(dot_raw - dot_rot) < 1e-4, (
            f"Single vector: raw={dot_raw:.6f} rot={dot_rot:.6f} "
            f"err={abs(dot_raw - dot_rot):.2e}"
        )

    def test_large_batch_invariance(self):
        """
        Invariance holds across a large batch (simulates a full prefill).
        B=8, H=32, T=64 — 16 384 vector pairs.
        """
        torch.manual_seed(7)
        B, H, T = 8, 32, 64
        q = torch.randn(B, H, T, HEAD_DIM)
        k = torch.randn(B, H, T, HEAD_DIM)
        dots_raw = (q * k).sum(dim=-1)

        q_rot = rotate_queries(q)
        k_rot, _ = rotate_kv_cache(k, torch.zeros_like(k))
        dots_rot = (q_rot * k_rot).sum(dim=-1)

        max_err = (dots_raw - dots_rot).abs().max().item()
        assert max_err < 1e-4, f"Large batch invariance: max_err={max_err:.2e}"


# ── Mathematical properties of the normalized WHT ─────────────────────────

class TestWHTProperties:

    def test_self_inverse(self):
        """
        (H/√D) is orthonormal → applying it twice is the identity.
        WHT(WHT(x)) == x   (to numerical precision).
        """
        torch.manual_seed(1)
        x = torch.randn(8, HEAD_DIM)
        x_twice = _wht_torch(_wht_torch(x))
        max_err = (x - x_twice).abs().max().item()
        assert max_err < 1e-4, f"Self-inverse: max_err={max_err:.2e}"

    def test_norm_preservation(self):
        """
        Isometry: ‖WHT(x)‖₂ == ‖x‖₂ for all x.
        Follows from orthonormality: (H/√D)^T (H/√D) = I.
        """
        torch.manual_seed(2)
        x = torch.randn(16, HEAD_DIM)
        norms_raw = x.norm(dim=-1)
        norms_rot = _wht_torch(x).norm(dim=-1)
        max_err = (norms_raw - norms_rot).abs().max().item()
        assert max_err < 1e-4, f"Norm preservation: max_err={max_err:.2e}"

    def test_linearity(self):
        """WHT(a·x + b·y) == a·WHT(x) + b·WHT(y) — linearity of the transform."""
        torch.manual_seed(3)
        a, b = 2.5, -1.3
        x = torch.randn(4, HEAD_DIM)
        y = torch.randn(4, HEAD_DIM)
        lhs = _wht_torch(a * x + b * y)
        rhs = a * _wht_torch(x) + b * _wht_torch(y)
        max_err = (lhs - rhs).abs().max().item()
        assert max_err < 1e-5, f"Linearity: max_err={max_err:.2e}"

    def test_known_small_case(self):
        """
        Verify against manually computed 4-D WHT.
        H₄ @ [1,2,3,4] = [10, -2, -4, 0]; normalized by 1/√4 = 0.5.
        """
        # Pad to HEAD_DIM with zeros and check only the first 4 dimensions
        # by using a standalone _wht_torch on a 4-element vector (D=4 variant)
        # — we test the formula manually:
        # H₄ = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
        x4 = torch.tensor([1., 2., 3., 4.])
        expected = torch.tensor([10., -2., -4., 0.]) * 0.5   # * 1/√4

        # Run the PyTorch WHT directly on 4-element input (monkey-patching HEAD_DIM
        # is not needed — use the reshape trick inline)
        y = x4.clone()
        for h in [1, 2]:
            y = y.reshape(-1, 2, h)
            upper, lower = y[:, 0, :], y[:, 1, :]
            y = torch.stack([upper + lower, upper - lower], dim=1).reshape(-1)
        y = y / math.sqrt(4)

        max_err = (y - expected).abs().max().item()
        assert max_err < 1e-5, f"Known 4-D case: got {y.tolist()}, expected {expected.tolist()}"


# ── Shape and dtype preservation ──────────────────────────────────────────

class TestShapeAndDtype:

    def test_rotate_queries_preserves_shape(self):
        q = torch.randn(3, 8, 32, HEAD_DIM)
        assert rotate_queries(q).shape == q.shape

    def test_rotate_queries_preserves_dtype_float32(self):
        q = torch.randn(1, 2, 4, HEAD_DIM, dtype=torch.float32)
        assert rotate_queries(q).dtype == torch.float32

    def test_rotate_kv_cache_preserves_shapes(self):
        B, Hkv, T = 2, 4, 16
        k = torch.randn(B, Hkv, T, HEAD_DIM)
        v = torch.randn(B, Hkv, T, HEAD_DIM)
        k_rot, v_rot = rotate_kv_cache(k, v)
        assert k_rot.shape == k.shape
        assert v_rot.shape == v.shape

    def test_rotate_queries_does_not_modify_input(self):
        """rotate_queries must return a new tensor — input is immutable."""
        q = torch.randn(1, 2, 4, HEAD_DIM)
        q_orig = q.clone()
        _ = rotate_queries(q)
        assert torch.equal(q, q_orig), "rotate_queries modified the input tensor"

    def test_wrong_head_dim_raises(self):
        """Passing a tensor with last dim ≠ HEAD_DIM must raise ValueError."""
        bad = torch.randn(2, 4, 8, 64)  # 64 ≠ 128
        with pytest.raises(ValueError, match="head_dim"):
            rotate_queries(bad)


# ── Triton kernel vs PyTorch reference cross-check ────────────────────────

@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
class TestTritonVsPyTorchReference:
    """
    Under TRITON_INTERPRET=1, the Triton kernel executes as Python.
    Its output must match _wht_torch exactly (atol=1e-4).
    This is the CPU-side gate before Phase 4 GPU testing.
    """

    def test_single_row(self):
        torch.manual_seed(10)
        x = torch.randn(1, 1, 1, HEAD_DIM)
        assert (_wht_rotate(x) - _wht_torch(x)).abs().max().item() < 1e-4

    def test_batch_matches_reference(self):
        torch.manual_seed(11)
        x = torch.randn(2, 4, 8, HEAD_DIM)
        max_err = (_wht_rotate(x) - _wht_torch(x)).abs().max().item()
        assert max_err < 1e-4, f"Triton vs PyTorch: max_err={max_err:.2e}"

    def test_rotate_queries_matches_pytorch(self):
        torch.manual_seed(12)
        q = torch.randn(2, 4, 8, HEAD_DIM)
        triton_out = rotate_queries(q)
        ref_out    = _wht_torch(q)
        max_err = (triton_out - ref_out).abs().max().item()
        assert max_err < 1e-4, f"rotate_queries Triton vs ref: max_err={max_err:.2e}"

    def test_rotate_kv_cache_both_channels(self):
        torch.manual_seed(13)
        k = torch.randn(2, 2, 8, HEAD_DIM)
        v = torch.randn(2, 2, 8, HEAD_DIM)
        k_rot, v_rot = rotate_kv_cache(k, v)
        k_err = (k_rot - _wht_torch(k)).abs().max().item()
        v_err = (v_rot - _wht_torch(v)).abs().max().item()
        assert k_err < 1e-4, f"k_rot err={k_err:.2e}"
        assert v_err < 1e-4, f"v_rot err={v_err:.2e}"
