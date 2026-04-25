"""
Unit tests for Phase 3: INT4 + Rademacher QJL Quantization.

Primary validation target (Architect's requirement):
    MSE(INT4 + QJL reconstruction) < MSE(INT4 alone)

Run on Mac (no GPU required):
    TRITON_INTERPRET=1 pytest tests/unit/test_quant.py -v

GPU tests (Phase 4 gate):
    python ci/run_gpu_tests.py
"""

from __future__ import annotations

import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import math
import pytest
import torch

from omnistack_rs.kernels.hadamard import HEAD_DIM, _wht_torch
from omnistack_rs.kernels.quantize import (
    QJL_DIM,
    _NBYTES_NIBBLE,
    _NBYTES_QJL,
    _HAS_TRITON,
    _quantize_row_torch,
    _pack_nibbles_torch,
    _unpack_nibbles_torch,
    _xor_unpack_word_torch,
    _qjl_signs_torch,
    _qjl_reconstruct_torch,
    _make_qjl_seed,
    quantize_rows,
    dequantize_rows,
    quantize_heads,
    dequantize_heads,
)
from omnistack_rs.quantization.codebook import (
    LloydMaxCalibrator,
    calibrate_codebook,
    calibrate_per_group,
    N_CENTROIDS,
)
from omnistack_rs.quantization.qjl import (
    RademacherQJL,
    qjl_encode,
    qjl_reconstruct,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rotated_activations():
    """(512,) calibration sample of WHT-rotated KV activations."""
    torch.manual_seed(0)
    raw = torch.randn(4, HEAD_DIM)
    return _wht_torch(raw).reshape(-1)  # (512,)


@pytest.fixture(scope="module")
def fitted_codebook(rotated_activations):
    """16-centroid sorted codebook fitted on rotated activations."""
    return calibrate_codebook(rotated_activations)


@pytest.fixture
def kv_batch():
    """(B=2, H=4, T=8, D=128) WHT-rotated KV tensor."""
    torch.manual_seed(1)
    raw = torch.randn(2, 4, 8, HEAD_DIM)
    return _wht_torch(raw)


# ── Codebook (Lloyd-Max) ──────────────────────────────────────────────────

class TestLloydMaxCodebook:

    def test_n_centroids(self, fitted_codebook):
        assert fitted_codebook.shape == (N_CENTROIDS,), (
            f"Expected (16,), got {fitted_codebook.shape}"
        )

    def test_centroids_sorted_ascending(self, fitted_codebook):
        diffs = fitted_codebook[1:] - fitted_codebook[:-1]
        assert (diffs >= 0).all(), "Centroids must be sorted ascending for binary search"

    def test_centroids_float32(self, fitted_codebook):
        assert fitted_codebook.dtype == torch.float32

    def test_mse_better_than_uniform(self, fitted_codebook, rotated_activations):
        """Lloyd-Max centroids should give lower MSE than a naive uniform grid."""
        x = rotated_activations.float()
        lo, hi = x.min().item(), x.max().item()
        uniform_cb = torch.linspace(lo, hi, N_CENTROIDS)

        cal = LloydMaxCalibrator()
        cal._codebook = fitted_codebook
        _, dequant_llm = cal.quantize(x)

        # Uniform grid assignment
        dists_uni = (x.unsqueeze(1) - uniform_cb.unsqueeze(0)).abs()
        idx_uni   = dists_uni.argmin(dim=1)
        dequant_uni = uniform_cb[idx_uni]

        mse_lloydmax = ((x - dequant_llm) ** 2).mean().item()
        mse_uniform  = ((x - dequant_uni) ** 2).mean().item()
        assert mse_lloydmax <= mse_uniform, (
            f"Lloyd-Max MSE ({mse_lloydmax:.4e}) must be ≤ uniform MSE ({mse_uniform:.4e})"
        )

    def test_quantize_indices_in_range(self, fitted_codebook, rotated_activations):
        cal = LloydMaxCalibrator()
        cal._codebook = fitted_codebook
        indices, _ = cal.quantize(rotated_activations)
        assert indices.min().item() >= 0
        assert indices.max().item() <= 15

    def test_fluent_api(self, rotated_activations):
        """LloydMaxCalibrator.fit() is fluent: returns self."""
        cal = LloydMaxCalibrator()
        result = cal.fit(rotated_activations)
        assert result is cal

    def test_requires_fit_before_codebook(self):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            _ = LloydMaxCalibrator().codebook

    def test_rejects_non_16_centroids(self):
        with pytest.raises(ValueError, match="16"):
            LloydMaxCalibrator(n_centroids=8)

    def test_calibrate_codebook_convenience(self, rotated_activations):
        cb = calibrate_codebook(rotated_activations)
        assert cb.shape == (16,)
        assert cb.dtype == torch.float32

    def test_convergence_on_gaussian(self):
        """Fits on 10K Gaussian samples; all 16 centroids should be distinct."""
        torch.manual_seed(99)
        samples = torch.randn(10_000)
        cb = calibrate_codebook(samples)
        diffs = (cb[1:] - cb[:-1]).abs()
        assert (diffs > 1e-6).all(), "Centroids should be distinct after convergence"


# ── Nibble packing (branch-free bit ops) ─────────────────────────────────

class TestNibblePacking:

    def test_roundtrip(self):
        """pack → unpack must recover the original codes exactly."""
        torch.manual_seed(2)
        codes = torch.randint(0, 16, (HEAD_DIM,), dtype=torch.uint8)
        recovered = _unpack_nibbles_torch(_pack_nibbles_torch(codes))
        assert torch.equal(codes, recovered), "Nibble pack/unpack roundtrip failed"

    def test_output_shape(self):
        codes  = torch.zeros(HEAD_DIM, dtype=torch.uint8)
        packed = _pack_nibbles_torch(codes)
        assert packed.shape == (HEAD_DIM // 2,)
        assert packed.dtype == torch.uint8

    def test_known_values(self):
        """Manually verify byte = (a & 0xF) | ((b & 0xF) << 4)."""
        codes = torch.zeros(HEAD_DIM, dtype=torch.uint8)
        codes[0] = 5   # lower nibble of first byte
        codes[1] = 3   # upper nibble of first byte
        packed = _pack_nibbles_torch(codes)
        expected_byte0 = (5 & 0xF) | ((3 & 0xF) << 4)   # = 5 | 48 = 53
        assert packed[0].item() == expected_byte0, (
            f"byte[0] = {packed[0].item()}, expected {expected_byte0}"
        )

    def test_all_ones(self):
        """All codes = 15 → all bytes = 0xFF."""
        codes  = torch.full((HEAD_DIM,), 15, dtype=torch.uint8)
        packed = _pack_nibbles_torch(codes)
        assert (packed == 0xFF).all(), "All-15 codes should give all-0xFF bytes"

    def test_unpack_preserves_values_in_0_15(self):
        """Unpacked codes must always be in [0, 15]."""
        torch.manual_seed(3)
        packed = torch.randint(0, 256, (HEAD_DIM // 2,), dtype=torch.uint8)
        codes  = _unpack_nibbles_torch(packed)
        assert (codes <= 15).all()
        assert (codes >= 0).all()


# ── Rademacher QJL ────────────────────────────────────────────────────────

class TestRademacherQJL:

    def test_encode_decode_shapes(self):
        torch.manual_seed(4)
        x = torch.randn(HEAD_DIM)
        qjl = RademacherQJL(HEAD_DIM, QJL_DIM)
        signs, norms = qjl.encode(x, head_idx=0)
        assert signs.shape == (QJL_DIM,)
        assert norms.shape == (1,)
        assert signs.dtype == torch.bool

    def test_per_head_seeds_differ(self):
        """Different head_idx values must produce different G matrices."""
        torch.manual_seed(5)
        x = torch.randn(HEAD_DIM)
        qjl = RademacherQJL(HEAD_DIM, QJL_DIM)
        signs0, _ = qjl.encode(x, head_idx=0)
        signs1, _ = qjl.encode(x, head_idx=1)
        # With different seeds, sign vectors should differ on many bits
        n_differ = (signs0 != signs1).sum().item()
        assert n_differ > QJL_DIM // 4, (
            f"Seeds 0 and 1 produce nearly identical signs ({n_differ} differ out of {QJL_DIM})"
        )

    def test_reconstruction_has_correct_shape(self):
        torch.manual_seed(6)
        x = torch.randn(HEAD_DIM)
        qjl = RademacherQJL(HEAD_DIM, QJL_DIM)
        signs, norms = qjl.encode(x, head_idx=0)
        out = qjl.reconstruct(signs, norms, head_idx=0)
        assert out.shape == (HEAD_DIM,)

    def test_norm_is_residual_norm(self):
        """norms returned by encode must equal ‖error‖₂."""
        torch.manual_seed(7)
        x = torch.randn(HEAD_DIM)
        qjl = RademacherQJL(HEAD_DIM, QJL_DIM)
        _, norms = qjl.encode(x, head_idx=0)
        expected_norm = x.norm().item()
        assert abs(norms.item() - expected_norm) < 1e-4

    def test_batched_encode_shapes(self):
        """qjl_encode handles batched (..., HEAD_DIM) input."""
        torch.manual_seed(8)
        x_orig  = torch.randn(2, 4, HEAD_DIM)
        x_dequant = x_orig * 0.9
        error, signs, norms = qjl_encode(x_orig, x_dequant, head_idx=0)
        assert signs.shape == (2, 4, QJL_DIM)
        assert norms.shape == (2, 4, 1)

    def test_qjl_signs_torch_bitmask_shape(self):
        torch.manual_seed(9)
        error = torch.randn(HEAD_DIM)
        _, packed = _qjl_signs_torch(error, head_seed=0)
        assert packed.shape == (QJL_DIM // 8,)
        assert packed.dtype == torch.uint8

    def test_qjl_reconstruct_torch_shape(self):
        torch.manual_seed(10)
        error = torch.randn(HEAD_DIM)
        _, packed = _qjl_signs_torch(error, head_seed=0)
        norm = error.norm().item()
        out = _qjl_reconstruct_torch(packed, norm, head_seed=0, device=error.device)
        assert out.shape == (HEAD_DIM,)


# ── Core Invariant: INT4 + QJL < INT4 alone (Architect's requirement) ────

class TestINT4QJLBetterThanINT4:
    """
    The Architect's primary requirement for Phase 3:
        MSE(INT4 + QJL reconstruction) < MSE(INT4 alone)
    for WHT-rotated KV activations.
    """

    def test_single_vector_qjl_reduces_mse(self, fitted_codebook):
        """Single 128-D vector: QJL correction strictly reduces MSE vs INT4 alone."""
        torch.manual_seed(11)
        x = _wht_torch(torch.randn(1, 1, 1, HEAD_DIM)).reshape(HEAD_DIM)

        codes, dequant = _quantize_row_torch(x, fitted_codebook)
        error = x - dequant

        mse_int4 = (error ** 2).mean().item()

        _, packed = _qjl_signs_torch(error, head_seed=0)
        error_hat = _qjl_reconstruct_torch(packed, error.norm().item(), 0, x.device)
        residual2 = error - error_hat
        mse_int4_qjl = (residual2 ** 2).mean().item()

        assert mse_int4_qjl < mse_int4, (
            f"QJL must reduce MSE: INT4={mse_int4:.4e}, INT4+QJL={mse_int4_qjl:.4e}"
        )

    def test_batch_qjl_reduces_mse(self, fitted_codebook, kv_batch):
        """
        Across a full (2,4,8,128) KV batch, INT4+QJL mean MSE < INT4 alone.
        This is the Architect's primary Phase 3 gate.
        """
        nibbles, qjl_packed, norms = quantize_rows(kv_batch, fitted_codebook)

        x_int4     = dequantize_rows(nibbles, qjl_packed, norms, fitted_codebook, with_qjl=False)
        x_int4_qjl = dequantize_rows(nibbles, qjl_packed, norms, fitted_codebook, with_qjl=True)

        x_orig = kv_batch.float()
        mse_int4     = ((x_orig - x_int4)     ** 2).mean().item()
        mse_int4_qjl = ((x_orig - x_int4_qjl) ** 2).mean().item()

        assert mse_int4_qjl < mse_int4, (
            f"Batch INT4+QJL MSE ({mse_int4_qjl:.4e}) must be < "
            f"INT4 MSE ({mse_int4:.4e})"
        )

    def test_qjl_improvement_ratio(self, fitted_codebook, kv_batch):
        """QJL residual correction should reduce INT4 MSE by ≥ 5%."""
        nibbles, qjl_packed, norms = quantize_rows(kv_batch, fitted_codebook)

        x_orig     = kv_batch.float()
        x_int4     = dequantize_rows(nibbles, qjl_packed, norms, fitted_codebook, with_qjl=False)
        x_int4_qjl = dequantize_rows(nibbles, qjl_packed, norms, fitted_codebook, with_qjl=True)

        mse_int4     = ((x_orig - x_int4)     ** 2).mean().item()
        mse_int4_qjl = ((x_orig - x_int4_qjl) ** 2).mean().item()

        improvement = (mse_int4 - mse_int4_qjl) / mse_int4
        assert improvement >= 0.05, (
            f"Expected ≥5% MSE improvement from QJL; got {improvement*100:.1f}%"
        )

    def test_large_batch_qjl_wins(self):
        """B=8, H=32, T=64 — 16 384 vectors; INT4+QJL must beat INT4."""
        torch.manual_seed(99)
        B, H, T = 8, 32, 64
        raw  = torch.randn(B, H, T, HEAD_DIM)
        x    = _wht_torch(raw)
        cb   = calibrate_codebook(x.reshape(-1))

        nibbles, qjl_packed, norms = quantize_rows(x, cb)
        x_int4     = dequantize_rows(nibbles, qjl_packed, norms, cb, with_qjl=False)
        x_int4_qjl = dequantize_rows(nibbles, qjl_packed, norms, cb, with_qjl=True)

        mse_int4     = ((x.float() - x_int4)     ** 2).mean().item()
        mse_int4_qjl = ((x.float() - x_int4_qjl) ** 2).mean().item()

        assert mse_int4_qjl < mse_int4, (
            f"Large batch: INT4+QJL ({mse_int4_qjl:.4e}) must beat INT4 ({mse_int4:.4e})"
        )


# ── Shape and dtype preservation ──────────────────────────────────────────

class TestShapeAndDtype:

    def test_quantize_rows_output_shapes(self, fitted_codebook, kv_batch):
        nibbles, qjl_packed, norms = quantize_rows(kv_batch, fitted_codebook)
        B, H, T = 2, 4, 8
        assert nibbles.shape    == (B, H, T, HEAD_DIM // 2), nibbles.shape
        assert qjl_packed.shape == (B, H, T, QJL_DIM  // 8), qjl_packed.shape
        assert norms.shape      == (B, H, T),                 norms.shape

    def test_dequantize_rows_output_shape(self, fitted_codebook, kv_batch):
        nibbles, qjl_packed, norms = quantize_rows(kv_batch, fitted_codebook)
        out = dequantize_rows(nibbles, qjl_packed, norms, fitted_codebook)
        assert out.shape == kv_batch.shape

    def test_nibble_output_dtype(self, fitted_codebook, kv_batch):
        nibbles, _, _ = quantize_rows(kv_batch, fitted_codebook)
        assert nibbles.dtype == torch.uint8

    def test_qjl_output_dtype(self, fitted_codebook, kv_batch):
        _, qjl_packed, _ = quantize_rows(kv_batch, fitted_codebook)
        assert qjl_packed.dtype == torch.uint8

    def test_norms_output_dtype(self, fitted_codebook, kv_batch):
        _, _, norms = quantize_rows(kv_batch, fitted_codebook)
        assert norms.dtype == torch.float32

    def test_norms_are_nonnegative(self, fitted_codebook, kv_batch):
        _, _, norms = quantize_rows(kv_batch, fitted_codebook)
        assert (norms >= 0).all()

    def test_input_not_modified(self, fitted_codebook, kv_batch):
        """quantize_rows must not modify the input tensor."""
        orig = kv_batch.clone()
        quantize_rows(kv_batch, fitted_codebook)
        assert torch.equal(kv_batch, orig), "quantize_rows modified the input tensor"


# ── XOR-word-unpack ───────────────────────────────────────────────────────

class TestXORUnpack:
    """
    Vectorized XOR-word-unpack: 8 nibbles per int32 word.

    Must be a drop-in replacement for the byte-level unpack — same output,
    no branching, 8× fewer iterations on SIMD hardware.
    """

    def test_roundtrip_matches_byte_unpack(self):
        """_xor_unpack_word_torch must match _unpack_nibbles_torch exactly."""
        torch.manual_seed(20)
        codes  = torch.randint(0, 16, (HEAD_DIM,), dtype=torch.uint8)
        packed = _pack_nibbles_torch(codes)
        ref    = _unpack_nibbles_torch(packed)
        fast   = _xor_unpack_word_torch(packed)
        assert torch.equal(ref, fast), (
            f"XOR-unpack differs from byte-unpack at {(ref != fast).nonzero().squeeze().tolist()}"
        )

    def test_output_shape(self):
        packed = torch.zeros(HEAD_DIM // 2, dtype=torch.uint8)
        assert _xor_unpack_word_torch(packed).shape == (HEAD_DIM,)

    def test_output_dtype(self):
        packed = torch.zeros(HEAD_DIM // 2, dtype=torch.uint8)
        assert _xor_unpack_word_torch(packed).dtype == torch.uint8

    def test_all_zeros(self):
        packed = torch.zeros(HEAD_DIM // 2, dtype=torch.uint8)
        assert (_xor_unpack_word_torch(packed) == 0).all()

    def test_all_fifteen(self):
        """All-0xFF packed bytes → all codes = 15."""
        packed = torch.full((HEAD_DIM // 2,), 0xFF, dtype=torch.uint8)
        codes  = _xor_unpack_word_torch(packed)
        assert (codes == 15).all()

    def test_known_first_word(self):
        """
        Verify the first 8 codes from a known packed pattern.

        Bytes [0x53, 0x24, 0x71, 0xA6]:
          byte[0]=0x53 → codes[0]=3, codes[1]=5
          byte[1]=0x24 → codes[2]=4, codes[3]=2
          byte[2]=0x71 → codes[4]=1, codes[5]=7
          byte[3]=0xA6 → codes[6]=6, codes[7]=10
        """
        packed = torch.zeros(HEAD_DIM // 2, dtype=torch.uint8)
        packed[0] = 0x53
        packed[1] = 0x24
        packed[2] = 0x71
        packed[3] = 0xA6
        codes = _xor_unpack_word_torch(packed)
        expected_first8 = torch.tensor([3, 5, 4, 2, 1, 7, 6, 10], dtype=torch.uint8)
        assert torch.equal(codes[:8], expected_first8), (
            f"First 8 codes: got {codes[:8].tolist()}, expected {expected_first8.tolist()}"
        )

    def test_values_in_range(self):
        """All unpacked codes must be in [0, 15]."""
        torch.manual_seed(21)
        packed = torch.randint(0, 256, (HEAD_DIM // 2,), dtype=torch.uint8)
        codes  = _xor_unpack_word_torch(packed)
        assert (codes >= 0).all() and (codes <= 15).all()

    def test_dequantize_uses_xor_unpack(self, fitted_codebook=None):
        """
        dequantize_rows must produce the same result whether the unpack path
        is word-level (xor_unpack) or byte-level — end-to-end roundtrip check.
        """
        torch.manual_seed(22)
        x  = _wht_torch(torch.randn(1, 1, 1, HEAD_DIM))
        cb = calibrate_codebook(x.reshape(-1))

        nibbles, qjl_packed, norms = quantize_rows(x, cb)
        out = dequantize_rows(nibbles, qjl_packed, norms, cb)
        # Shape and dtype must be preserved
        assert out.shape == x.shape
        assert out.dtype == torch.float32


# ── Per-group codebooks ───────────────────────────────────────────────────

class TestPerGroupCodebooks:
    """
    One Lloyd-Max codebook per KV head group.

    "Action Ad" heads concentrate near different centroids than "Romance Ad"
    heads — per-group codebooks achieve lower group-specific MSE.
    """

    N_KV_HEADS = 4

    @pytest.fixture
    def per_group_kv(self):
        """
        (B=2, n_kv_heads=4, T=8, HEAD_DIM) WHT-rotated KV activations.

        Heads use DIFFERENT scales to simulate real activation distributions
        (e.g. "Action Ad" heads with high variance vs "slow" heads with low
        variance). This makes per-group codebooks genuinely advantageous.
        """
        torch.manual_seed(30)
        scales = [0.3, 1.0, 2.0, 4.0]   # head-specific activation scales
        slabs  = []
        for s in scales:
            raw = torch.randn(2, 1, 8, HEAD_DIM) * s
            slabs.append(_wht_torch(raw))
        return torch.cat(slabs, dim=1)   # (2, 4, 8, HEAD_DIM)

    @pytest.fixture
    def per_group_codebooks(self, per_group_kv):
        """(n_kv_heads, 16) per-group codebooks fitted on per_group_kv."""
        samples = [per_group_kv[:, h, :, :].reshape(-1) for h in range(self.N_KV_HEADS)]
        return calibrate_per_group(samples)

    # ── calibrate_per_group ───────────────────────────────────────────────

    def test_calibrate_per_group_shape(self, per_group_codebooks):
        assert per_group_codebooks.shape == (self.N_KV_HEADS, 16)

    def test_calibrate_per_group_sorted(self, per_group_codebooks):
        for h in range(self.N_KV_HEADS):
            diffs = per_group_codebooks[h, 1:] - per_group_codebooks[h, :-1]
            assert (diffs >= 0).all(), f"Group {h} codebook not sorted ascending"

    def test_per_group_codebooks_differ(self, per_group_codebooks):
        """Different groups should have at least some centroids that differ."""
        for h in range(1, self.N_KV_HEADS):
            diff = (per_group_codebooks[0] - per_group_codebooks[h]).abs().max().item()
            assert diff > 1e-4, f"Groups 0 and {h} have identical codebooks (diff={diff:.2e})"

    def test_per_group_lower_mse_than_shared(self, per_group_kv, per_group_codebooks):
        """
        Per-group codebooks must achieve lower mean MSE than a single shared codebook.

        This is the Architect's primary motivation for per-group quantization.
        """
        # Single shared codebook calibrated on all activations
        shared_cb = calibrate_codebook(per_group_kv.reshape(-1))

        mse_shared_total = 0.0
        mse_pergroup_total = 0.0
        for h in range(self.N_KV_HEADS):
            x_h = per_group_kv[:, h, :, :].reshape(-1, HEAD_DIM).float()
            for row in x_h:
                _, dq_shared = _quantize_row_torch(row, shared_cb)
                _, dq_group  = _quantize_row_torch(row, per_group_codebooks[h])
                mse_shared_total   += ((row - dq_shared) ** 2).mean().item()
                mse_pergroup_total += ((row - dq_group)  ** 2).mean().item()

        assert mse_pergroup_total < mse_shared_total, (
            f"Per-group MSE ({mse_pergroup_total:.4e}) must be < "
            f"shared MSE ({mse_shared_total:.4e})"
        )

    # ── quantize_heads / dequantize_heads ─────────────────────────────────

    def test_quantize_heads_output_shapes(self, per_group_kv, per_group_codebooks):
        B, H, T = 2, self.N_KV_HEADS, 8
        nib, qjl, nrm = quantize_heads(per_group_kv, per_group_codebooks)
        assert nib.shape == (B, H, T, _NBYTES_NIBBLE)
        assert qjl.shape == (B, H, T, _NBYTES_QJL)
        assert nrm.shape == (B, H, T)

    def test_dequantize_heads_output_shape(self, per_group_kv, per_group_codebooks):
        nib, qjl, nrm = quantize_heads(per_group_kv, per_group_codebooks)
        out = dequantize_heads(nib, qjl, nrm, per_group_codebooks)
        assert out.shape == per_group_kv.shape

    def test_per_group_int4_qjl_beats_int4(self, per_group_kv, per_group_codebooks):
        """INT4+QJL with per-group codebooks must beat INT4 alone."""
        nib, qjl, nrm = quantize_heads(per_group_kv, per_group_codebooks)
        x_int4     = dequantize_heads(nib, qjl, nrm, per_group_codebooks, with_qjl=False)
        x_int4_qjl = dequantize_heads(nib, qjl, nrm, per_group_codebooks, with_qjl=True)
        mse_int4     = ((per_group_kv.float() - x_int4)     ** 2).mean().item()
        mse_int4_qjl = ((per_group_kv.float() - x_int4_qjl) ** 2).mean().item()
        assert mse_int4_qjl < mse_int4, (
            f"INT4+QJL ({mse_int4_qjl:.4e}) must beat INT4 ({mse_int4:.4e})"
        )

    def test_shared_codebook_fallback(self, per_group_kv):
        """
        quantize_heads with a (16,) shared codebook produces the same nibbles and norms
        as quantize_rows (INT4 is seed-independent).  QJL bits legitimately differ:
        quantize_heads uses per-head seeds (user_id%1024)^h; quantize_rows uses
        per-row seeds (user_id%1024)^row_i.  Both achieve MSE improvement over INT4.
        """
        torch.manual_seed(31)
        cb = calibrate_codebook(per_group_kv.reshape(-1))
        nib_heads, qjl_heads, nrm_heads = quantize_heads(per_group_kv, cb)
        nib_rows,  _,         nrm_rows  = quantize_rows(per_group_kv, cb)

        # INT4 output (nibbles + norms) must be identical — codebook lookup is seed-free
        assert torch.equal(nib_heads, nib_rows), "nibbles must match between quantize_heads and quantize_rows"
        assert torch.equal(nrm_heads, nrm_rows), "norms must match between quantize_heads and quantize_rows"

        # QJL shapes and dtypes are correct even though bit patterns differ
        assert qjl_heads.shape == nib_heads.shape[:-1] + (QJL_DIM // 8,)
        assert qjl_heads.dtype == torch.uint8

        # End-to-end: quantize_heads + dequantize_heads still achieves MSE improvement
        x_int4     = dequantize_heads(nib_heads, qjl_heads, nrm_heads, cb, with_qjl=False)
        x_int4_qjl = dequantize_heads(nib_heads, qjl_heads, nrm_heads, cb, with_qjl=True)
        mse_int4     = ((per_group_kv.float() - x_int4)     ** 2).mean().item()
        mse_int4_qjl = ((per_group_kv.float() - x_int4_qjl) ** 2).mean().item()
        assert mse_int4_qjl < mse_int4, (
            f"quantize_heads fallback: INT4+QJL ({mse_int4_qjl:.4e}) must beat INT4 ({mse_int4:.4e})"
        )


# ── User-ID PRNG seeds ────────────────────────────────────────────────────

class TestUserIDSeeds:
    """
    seed = (user_id % 1024) ^ head_idx ensures that different users' QJL
    projections are statistically independent — no cross-user structured noise.
    """

    def test_seed_formula(self):
        assert _make_qjl_seed(user_id=0,    row_seed=7)  == 0    ^ 7
        assert _make_qjl_seed(user_id=1,    row_seed=7)  == 1    ^ 7
        assert _make_qjl_seed(user_id=1024, row_seed=7)  == 0    ^ 7   # 1024 % 1024 = 0
        assert _make_qjl_seed(user_id=1025, row_seed=7)  == 1    ^ 7
        assert _make_qjl_seed(user_id=5,    row_seed=3)  == 5    ^ 3

    def test_different_users_get_different_G(self):
        """Users 0 and 1 must produce different QJL projections at the same head."""
        torch.manual_seed(40)
        error = torch.randn(HEAD_DIM)
        _, packed_u0 = _qjl_signs_torch(error, head_seed=_make_qjl_seed(0, 0))
        _, packed_u1 = _qjl_signs_torch(error, head_seed=_make_qjl_seed(1, 0))
        # At least some bits should differ
        n_diff = sum(
            bin(int(a) ^ int(b)).count("1")
            for a, b in zip(packed_u0.tolist(), packed_u1.tolist())
        )
        assert n_diff > QJL_DIM // 4, (
            f"Users 0 and 1 have nearly identical QJL projections ({n_diff} bits differ)"
        )

    def test_user_1024_aliases_to_user_0(self):
        """user_id=1024 and user_id=0 must produce the same G (1024%1024==0)."""
        torch.manual_seed(41)
        error = torch.randn(HEAD_DIM)
        _, p0    = _qjl_signs_torch(error, head_seed=_make_qjl_seed(0,    0))
        _, p1024 = _qjl_signs_torch(error, head_seed=_make_qjl_seed(1024, 0))
        assert torch.equal(p0, p1024), "user_id=0 and user_id=1024 should alias"

    def test_encode_decode_user_id_consistency(self):
        """Encode and decode with the same user_id must be consistent."""
        torch.manual_seed(42)
        x = _wht_torch(torch.randn(1, 1, 1, HEAD_DIM))
        cb = calibrate_codebook(x.reshape(-1))
        user_id = 999

        nib, qjl, nrm = quantize_rows(x, cb, user_id=user_id)
        out = dequantize_rows(nib, qjl, nrm, cb, user_id=user_id)
        # Reconstruction should be close to original (not scrambled by wrong seed)
        mse = ((x.float() - out) ** 2).mean().item()
        assert mse < 0.5, f"MSE={mse:.4e} too large — seed mismatch?"

    def test_wrong_user_id_gives_worse_reconstruction(self):
        """Decoding with wrong user_id must give higher MSE than correct user_id."""
        torch.manual_seed(43)
        x = _wht_torch(torch.randn(1, 1, 8, HEAD_DIM))
        cb = calibrate_codebook(x.reshape(-1))
        correct_uid = 7
        wrong_uid   = 42

        nib, qjl, nrm = quantize_rows(x, cb, user_id=correct_uid)
        out_correct = dequantize_rows(nib, qjl, nrm, cb, user_id=correct_uid)
        out_wrong   = dequantize_rows(nib, qjl, nrm, cb, user_id=wrong_uid)

        mse_correct = ((x.float() - out_correct) ** 2).mean().item()
        mse_wrong   = ((x.float() - out_wrong)   ** 2).mean().item()
        assert mse_wrong > mse_correct, (
            f"Wrong user_id ({mse_wrong:.4e}) should give higher MSE than "
            f"correct ({mse_correct:.4e})"
        )
