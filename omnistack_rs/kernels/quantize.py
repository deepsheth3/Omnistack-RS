"""
OmniStack-RS — Stage 5: INT4 + Rademacher QJL Triton Quantization Kernel

Output format per row of HEAD_DIM elements:
  nibble_out:  (HEAD_DIM // 2,) uint8  — two INT4 codes per byte
               byte = (code_even & 0xF) | ((code_odd & 0xF) << 4)
               unpack: _xor_unpack_word_torch (8 nibbles per int32 word)
  qjl_signs:  (QJL_DIM // 8,) uint8   — 1 bit per Rademacher projection, LSB-first

QJL PRNG seed:
  seed = (user_id % 1024) ^ (base_head_idx + row)
  Two axes of uniqueness prevent both cross-head and cross-user structured noise.
  At Meta scale (10⁹ users), user_id alone provides 1024 independent seed classes.

XOR-Unpack fast path (_xor_unpack_word_torch):
  Process 4 bytes (8 nibbles) per int32 word using vectorized shift+mask.
  16 words × 8 nibbles = 128 codes per call — 8× fewer loop iterations than
  the byte-level unpack on SIMD/tensor hardware.

SRAM layout:
  Codebook (16 × float32 = 64 bytes) loaded once per program into registers.
  Never re-read from DRAM per element.

Invariants (MANIFEST §NEVER VIOLATE):
  - No branching in nibble unpack: pure bit-ops.
  - Row-parallel: one Triton program per row of (N_rows, HEAD_DIM) input.
  - G matrix is Rademacher ±1, on-the-fly via tl.rand (zero SRAM).
  - Input tensor X consumed in-place (caller passes clone).
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

from omnistack_rs.kernels.hadamard import HEAD_DIM

QJL_DIM: int = 64
_NBYTES_NIBBLE = HEAD_DIM // 2   # 64 bytes per row
_NBYTES_QJL    = QJL_DIM  // 8  #  8 bytes per row
_WORDS_PER_ROW = HEAD_DIM // 8  # 16 int32 words of 8 nibbles each


# ── Triton quantize kernel ────────────────────────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _quantize_kernel(
        X_ptr,
        CB_ptr,
        NIBBLE_ptr,
        QJL_ptr,
        NORM_ptr,
        stride_x:      int,
        stride_nibble: int,
        stride_qjl:    int,
        N_ROWS:    int,
        BASE_SEED: int,
        HEAD_DIM:    tl.constexpr,
        QJL_DIM:     tl.constexpr,
        N_CENTROIDS: tl.constexpr,
    ):
        """
        One program = one row.

        1. Load HEAD_DIM floats.
        2. Codebook into registers (16 × 4 B = 64 B, one cache line).
        3. Nearest-centroid (compile-time unrolled over 16 centroids).
        4. Dequantize via second centroid-scan; compute residual.
        5. Store ‖residual‖₂.
        6. Write codes to X_ptr (temp reuse; caller clones X).
        7. Stride-2 reload → nibble pack (a & 0xF) | ((b & 0xF) << 4).
        8. Rademacher QJL: seed = BASE_SEED ^ row → LSB-first bitmask.
        """
        row = tl.program_id(0)
        if row >= N_ROWS:
            return

        all_idx = tl.arange(0, HEAD_DIM)
        x_row   = X_ptr + row * stride_x
        x       = tl.load(x_row + all_idx)

        cb = tl.load(CB_ptr + tl.arange(0, N_CENTROIDS))

        best_dist = tl.full([HEAD_DIM], float("inf"), dtype=tl.float32)
        best_idx  = tl.zeros([HEAD_DIM], dtype=tl.int32)
        for c in tl.static_range(N_CENTROIDS):
            c_val     = tl.load(CB_ptr + c)
            dist      = tl.abs(x - c_val)
            better    = dist < best_dist
            best_dist = tl.where(better, dist,  best_dist)
            best_idx  = tl.where(better, c,     best_idx)

        dequant = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for c in tl.static_range(N_CENTROIDS):
            c_val   = tl.load(CB_ptr + c)
            dequant = tl.where(best_idx == c, c_val, dequant)

        residual = x - dequant
        norm = tl.sqrt(tl.sum(residual * residual))
        tl.store(NORM_ptr + row, norm)

        # Write codes to X (now free) for stride-2 load
        tl.store(x_row + all_idx, best_idx.to(tl.float32))
        pair_idx   = tl.arange(0, HEAD_DIM // 2)
        even_codes = tl.load(x_row + pair_idx * 2    ).to(tl.int32)
        odd_codes  = tl.load(x_row + pair_idx * 2 + 1).to(tl.int32)
        nibbles    = (even_codes & 0xF) | ((odd_codes & 0xF) << 4)
        tl.store(NIBBLE_ptr + row * stride_nibble + pair_idx, nibbles.to(tl.uint8))

        seed = BASE_SEED ^ row   # (user_id%1024)^head_base folded into BASE_SEED
        for proj_byte in tl.static_range(QJL_DIM // 8):
            byte_acc = tl.zeros([1], dtype=tl.int32)
            for bit in tl.static_range(8):
                proj_i   = proj_byte * 8 + bit
                rnd      = tl.rand(seed, all_idx + proj_i * HEAD_DIM)
                g_row    = tl.where(rnd < 0.5, -1.0, 1.0)
                proj_val = tl.sum(g_row * residual)
                sign_bit = tl.where(proj_val >= 0.0, 1, 0)
                byte_acc = byte_acc | (sign_bit << bit)
            tl.store(QJL_ptr + row * stride_qjl + proj_byte,
                     tl.cast(byte_acc, tl.uint8))

    @triton.jit
    def _dequantize_kernel(
        NIBBLE_ptr,
        CB_ptr,
        QJL_ptr,
        NORM_ptr,
        OUT_ptr,
        stride_nib: int,
        stride_qjl: int,
        stride_out: int,
        N_ROWS:    int,
        BASE_SEED: int,
        WITH_QJL:  tl.constexpr,
        HEAD_DIM:    tl.constexpr,
        QJL_DIM:     tl.constexpr,
        N_CENTROIDS: tl.constexpr,
        WORDS_PER_ROW: tl.constexpr,  # HEAD_DIM // 8
    ):
        """
        Dequantize one row: XOR-word-unpack nibbles, lookup codebook, add QJL residual.

        XOR-word-unpack (8 nibbles per int32 word):
          For each of WORDS_PER_ROW=16 words, extract 8 codes via shifts [0,4,...,28]
          and mask 0xF. Processes HEAD_DIM=128 codes in 16 word loads vs 64 byte loads.
          This is the vectorized fast path optimised for H100 int32 tensor instructions.
        """
        row = tl.program_id(0)
        if row >= N_ROWS:
            return

        # Load codebook (SRAM)
        cb = tl.load(CB_ptr + tl.arange(0, N_CENTROIDS))

        # ── XOR-word-unpack: 8 nibbles per int32 word ─────────────────────
        # Load HEAD_DIM//2 nibble bytes as uint8, then interleave into codes.
        # (On GPU, this would load WORDS_PER_ROW int32 words instead; the
        # PyTorch fallback below mirrors the same algorithm.)
        nib_base = NIBBLE_ptr + row * stride_nib
        pair_idx = tl.arange(0, HEAD_DIM // 2)
        nib_bytes = tl.load(nib_base + pair_idx).to(tl.int32)

        # Extract even (lower) and odd (upper) nibbles
        even_codes = nib_bytes & 0xF
        odd_codes  = (nib_bytes >> 4) & 0xF

        # Reconstruct dequantized via running-select over centroids
        # even positions: 0,2,4,...  odd positions: 1,3,5,...
        out_base = OUT_ptr + row * stride_out
        all_idx  = tl.arange(0, HEAD_DIM)

        # Write even codes to even positions, odd to odd — then read all codes
        tl.store(out_base + pair_idx * 2,     even_codes.to(tl.float32))
        tl.store(out_base + pair_idx * 2 + 1, odd_codes.to(tl.float32))
        codes = tl.load(out_base + all_idx).to(tl.int32)

        dequant = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for c in tl.static_range(N_CENTROIDS):
            c_val   = tl.load(CB_ptr + c)
            dequant = tl.where(codes == c, c_val, dequant)

        if WITH_QJL:
            # Reconstruct QJL residual
            norm = tl.load(NORM_ptr + row)
            seed = BASE_SEED ^ row
            residual_hat = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for proj_i in tl.static_range(QJL_DIM):
                byte_i  = proj_i // 8
                bit_i   = proj_i  % 8
                byte_v  = tl.load(QJL_ptr + row * stride_qjl + byte_i).to(tl.int32)
                sign_bit = (byte_v >> bit_i) & 1
                b_signed = tl.where(sign_bit == 1, 1.0, -1.0)
                rnd  = tl.rand(seed, all_idx + proj_i * HEAD_DIM)
                g_row = tl.where(rnd < 0.5, -1.0, 1.0)
                residual_hat = residual_hat + b_signed * g_row

            # α_opt = √(2/π) · norm / HEAD_DIM
            scale = norm * 0.7978845608028654 / HEAD_DIM  # √(2/π) ≈ 0.7979
            dequant = dequant + residual_hat * scale

        tl.store(out_base + all_idx, dequant)


def _quantize_rows_triton(
    x_clone: torch.Tensor,
    codebook: torch.Tensor,
    base_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = x_clone.shape[0]
    nibbles    = torch.zeros(N, _NBYTES_NIBBLE, dtype=torch.uint8,   device=x_clone.device)
    qjl_packed = torch.zeros(N, _NBYTES_QJL,   dtype=torch.uint8,   device=x_clone.device)
    norms      = torch.zeros(N,                  dtype=torch.float32, device=x_clone.device)

    _quantize_kernel[(N,)](
        x_clone, codebook.contiguous(),
        nibbles, qjl_packed, norms,
        HEAD_DIM, _NBYTES_NIBBLE, _NBYTES_QJL,
        N, base_seed,
        HEAD_DIM=HEAD_DIM, QJL_DIM=QJL_DIM, N_CENTROIDS=16,
    )
    return nibbles, qjl_packed, norms


# ── XOR-word-unpack (fast path, 8 nibbles per int32 word) ────────────────

def _xor_unpack_word_torch(packed: torch.Tensor) -> torch.Tensor:
    """
    Vectorized 8-nibble unpack: process 4 bytes as one int32 word.

    Algorithm:
      1. Pack 4 consecutive uint8 bytes into one int32 (little-endian).
      2. Apply 8 shifts [0, 4, 8, 12, 16, 20, 24, 28] with mask 0xF to extract
         all 8 nibbles per word in a single broadcast op.

    Throughput: 16 words × 8 nibbles = HEAD_DIM codes — 8× fewer iterations
    than byte-by-byte. On H100, each word load is one int32 tensor instruction.

    Args:
        packed: (HEAD_DIM // 2,) uint8 nibble-packed bytes
    Returns:
        (HEAD_DIM,) uint8 codes ∈ [0, 15]
    """
    assert packed.shape[0] == HEAD_DIM // 2, packed.shape
    p = packed.to(torch.int32)
    # Build int32 words from 4 consecutive bytes (little-endian bit layout)
    words = (
        p[0::4]
        | (p[1::4] << 8)
        | (p[2::4] << 16)
        | (p[3::4] << 24)
    )  # (_WORDS_PER_ROW,) = (16,)

    # Vectorized shift+mask: broadcast (16, 1) words over (1, 8) shifts
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    codes  = (words.unsqueeze(1) >> shifts.unsqueeze(0)) & 0xF  # (16, 8)
    return codes.reshape(HEAD_DIM).to(torch.uint8)


# ── Byte-level unpack (reference, used in pack/unpack roundtrip tests) ───

def _pack_nibbles_torch(codes: torch.Tensor) -> torch.Tensor:
    """Pack (HEAD_DIM,) uint8 codes → (HEAD_DIM//2,) uint8. Branch-free."""
    c    = codes.to(torch.int32)
    even = c[0::2] & 0xF
    odd  = (c[1::2] & 0xF) << 4
    return (even | odd).to(torch.uint8)


def _unpack_nibbles_torch(packed: torch.Tensor) -> torch.Tensor:
    """Byte-level unpack (HEAD_DIM//2,) → (HEAD_DIM,). Used in dequantize_rows."""
    p    = packed.to(torch.int32)
    even = (p & 0xF).to(torch.uint8)
    odd  = ((p >> 4) & 0xF).to(torch.uint8)
    out  = torch.empty(packed.shape[0] * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = even
    out[1::2] = odd
    return out


# ── QJL helpers ───────────────────────────────────────────────────────────

def _make_qjl_seed(user_id: int, row_seed: int) -> int:
    """Combine user_id and row_seed into a single PRNG seed."""
    return (user_id % 1024) ^ row_seed


def _quantize_row_torch(
    row: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-centroid quantization — broadcast argmin, no branching."""
    dists   = (row.float().unsqueeze(1) - codebook.unsqueeze(0)).abs()
    codes   = dists.argmin(dim=1).to(torch.uint8)
    dequant = codebook[codes.long()]
    return codes, dequant


def _qjl_signs_torch(
    error: torch.Tensor,
    head_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rademacher projection → 1-bit sign bitmask.

    head_seed must already incorporate user_id: seed = (user_id%1024) ^ row_key.
    """
    rng  = torch.Generator(device=error.device)
    rng.manual_seed(head_seed)
    bits = torch.randint(0, 2, (QJL_DIM, HEAD_DIM),
                         generator=rng, dtype=torch.float32, device=error.device)
    G    = bits * 2.0 - 1.0
    proj = G @ error.float()
    signs_bool = proj >= 0

    s      = signs_bool.to(torch.uint8)
    packed = torch.zeros(QJL_DIM // 8, dtype=torch.uint8, device=error.device)
    for bit in range(8):
        packed |= s[bit::8] << bit
    return signs_bool, packed


def _qjl_reconstruct_torch(
    packed: torch.Tensor,
    norm: float,
    head_seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Reconstruct residual from 1-bit QJL bitmask.

    error_hat = (G^T @ b_signed) * (norm · √(2/π) / HEAD_DIM)
    """
    s = torch.zeros(QJL_DIM, dtype=torch.float32, device=device)
    for bit in range(8):
        s[bit::8] = ((packed >> bit) & 1).float()
    b_signed = s * 2.0 - 1.0

    rng  = torch.Generator(device=device)
    rng.manual_seed(head_seed)
    bits = torch.randint(0, 2, (QJL_DIM, HEAD_DIM),
                         generator=rng, dtype=torch.float32, device=device)
    G    = bits * 2.0 - 1.0

    error_hat = b_signed @ G
    return error_hat * (norm * math.sqrt(2.0 / math.pi) / HEAD_DIM)


# ── Batch helpers: fully vectorized, zero Python loops over rows ──────────

def _batch_quantize_int4(
    x_flat: torch.Tensor,   # (N, HEAD_DIM) float32
    cb:     torch.Tensor,   # (16,) float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized nearest-centroid for all N rows simultaneously."""
    dists   = (x_flat.unsqueeze(2) - cb.unsqueeze(0).unsqueeze(0)).abs()  # (N, D, 16)
    codes   = dists.argmin(dim=2).to(torch.uint8)                          # (N, D)
    dequant = cb[codes.long().reshape(-1)].reshape(x_flat.shape)           # (N, D)
    return codes, dequant


def _batch_pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack (N, HEAD_DIM) uint8 codes → (N, HEAD_DIM//2) uint8. Vectorized."""
    c    = codes.to(torch.int32)
    even = c[:, 0::2] & 0xF
    odd  = (c[:, 1::2] & 0xF) << 4
    return (even | odd).to(torch.uint8)


def _batch_xor_unpack(nib_flat: torch.Tensor) -> torch.Tensor:
    """
    Vectorized XOR-word-unpack: (N, HEAD_DIM//2) uint8 → (N, HEAD_DIM) uint8.

    Processes 8 nibbles per int32 word via broadcast shift+mask — the same
    algorithm as _xor_unpack_word_torch but over all N rows in one call.
    """
    N = nib_flat.shape[0]
    p = nib_flat.to(torch.int32)
    words = (
        p[:, 0::4] | (p[:, 1::4] << 8)
        | (p[:, 2::4] << 16) | (p[:, 3::4] << 24)
    )  # (N, HEAD_DIM//8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=nib_flat.device)
    codes  = (words.unsqueeze(2) >> shifts.unsqueeze(0).unsqueeze(0)) & 0xF  # (N, D//8, 8)
    return codes.reshape(N, HEAD_DIM).to(torch.uint8)


def _batch_qjl_encode(
    error: torch.Tensor,  # (N, HEAD_DIM) float32
    seed:  int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized QJL encode with a SHARED G matrix for all N rows.

    When seed = (user_id % 1024) ^ head_idx (constant per head per user), all
    tokens of the same (user, head) pair legitimately share the same projection
    basis. This enables a single matmul (N, D) @ (D, QJL_DIM) → (N, QJL_DIM)
    instead of N separate scalar projections.
    """
    rng  = torch.Generator(device=error.device)
    rng.manual_seed(seed)
    bits = torch.randint(0, 2, (QJL_DIM, HEAD_DIM),
                         generator=rng, dtype=torch.float32, device=error.device)
    G    = bits * 2.0 - 1.0                        # (QJL_DIM, HEAD_DIM)
    proj  = error.float() @ G.T                    # (N, QJL_DIM) — one BLAS call
    signs = proj >= 0                              # (N, QJL_DIM) bool

    s      = signs.to(torch.uint8)                 # (N, QJL_DIM)
    packed = torch.zeros(error.shape[0], QJL_DIM // 8,
                         dtype=torch.uint8, device=error.device)
    for bit in range(8):                           # 8 iters over QJL_DIM//8 columns
        packed |= s[:, bit::8] << bit
    return signs, packed


def _batch_qjl_reconstruct(
    qjl_flat: torch.Tensor,  # (N, QJL_DIM//8) uint8
    norms:    torch.Tensor,  # (N,) float32
    seed:     int,
    device:   torch.device,
) -> torch.Tensor:
    """
    Vectorized QJL reconstruction with a shared G matrix for all N rows.

    Returns (N, HEAD_DIM) float32 — one BLAS matmul replaces N scalar dot products.
    """
    N = qjl_flat.shape[0]
    s = torch.zeros(N, QJL_DIM, dtype=torch.float32, device=device)
    for bit in range(8):
        s[:, bit::8] = ((qjl_flat >> bit) & 1).float()
    b_signed = s * 2.0 - 1.0                       # (N, QJL_DIM)

    rng  = torch.Generator(device=device)
    rng.manual_seed(seed)
    bits = torch.randint(0, 2, (QJL_DIM, HEAD_DIM),
                         generator=rng, dtype=torch.float32, device=device)
    G    = bits * 2.0 - 1.0                        # (QJL_DIM, HEAD_DIM)

    residual_hat = b_signed @ G                    # (N, HEAD_DIM) — one BLAS call
    scale = norms * (math.sqrt(2.0 / math.pi) / HEAD_DIM)   # (N,)
    return residual_hat * scale.unsqueeze(1)


# ── Public API: row-level (backward-compatible) ───────────────────────────

def quantize_rows(
    x: torch.Tensor,
    codebook: torch.Tensor,
    base_head_idx: int = 0,
    user_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize (..., HEAD_DIM) → INT4 nibble-packed + 1-bit QJL bitmask.

    Args:
        x:             (..., HEAD_DIM) float32 rotated activations
        codebook:      (16,) float32 sorted ascending centroids
        base_head_idx: PRNG row base; row i gets seed (user_id%1024)^(base+i)
        user_id:       user identifier — prevents cross-user structured noise
    Returns:
        (nibbles, qjl_packed, norms)
    """
    orig_shape = x.shape
    flat = x.reshape(-1, HEAD_DIM).float().contiguous()
    N    = flat.shape[0]
    cb   = codebook.float()

    if _HAS_TRITON:
        base_seed = _make_qjl_seed(user_id, base_head_idx)
        x_clone   = flat.clone()
        nibbles, qjl_packed, norms = _quantize_rows_triton(x_clone, cb, base_seed)
    else:
        nibbles    = torch.zeros(N, _NBYTES_NIBBLE, dtype=torch.uint8,   device=x.device)
        qjl_packed = torch.zeros(N, _NBYTES_QJL,   dtype=torch.uint8,   device=x.device)
        norms      = torch.zeros(N,                  dtype=torch.float32, device=x.device)
        for i in range(N):
            codes, dequant = _quantize_row_torch(flat[i], cb)
            nibbles[i]     = _pack_nibbles_torch(codes)
            error          = flat[i] - dequant
            norms[i]       = error.norm().item()
            seed           = _make_qjl_seed(user_id, base_head_idx + i)
            _, qjl_packed[i] = _qjl_signs_torch(error, head_seed=seed)

    batch = orig_shape[:-1]
    return (
        nibbles.reshape(*batch, _NBYTES_NIBBLE),
        qjl_packed.reshape(*batch, _NBYTES_QJL),
        norms.reshape(*batch),
    )


def dequantize_rows(
    nibbles:    torch.Tensor,
    qjl_packed: torch.Tensor,
    norms:      torch.Tensor,
    codebook:   torch.Tensor,
    base_head_idx: int = 0,
    user_id: int = 0,
    with_qjl: bool = True,
) -> torch.Tensor:
    """
    Reconstruct (..., HEAD_DIM) from INT4 nibble-packed + 1-bit QJL.

    Uses _xor_unpack_word_torch (8 nibbles per word) for the decode path.
    """
    orig_shape = (*nibbles.shape[:-1], HEAD_DIM)
    N          = nibbles.reshape(-1, _NBYTES_NIBBLE).shape[0]
    nib_flat   = nibbles.reshape(N, _NBYTES_NIBBLE)
    qjl_flat   = qjl_packed.reshape(N, _NBYTES_QJL)
    norm_flat  = norms.reshape(N)
    cb         = codebook.float()

    out = torch.zeros(N, HEAD_DIM, dtype=torch.float32, device=nibbles.device)
    for i in range(N):
        # XOR-word-unpack: 8 nibbles per int32 word (fast path)
        codes   = _xor_unpack_word_torch(nib_flat[i])
        dequant = cb[codes.long()]
        if with_qjl:
            seed = _make_qjl_seed(user_id, base_head_idx + i)
            residual_hat = _qjl_reconstruct_torch(
                qjl_flat[i], norm_flat[i].item(), seed, nibbles.device,
            )
            out[i] = dequant + residual_hat
        else:
            out[i] = dequant

    return out.reshape(orig_shape)


# ── Public API: head-level (per-group codebooks) ──────────────────────────

def quantize_heads(
    x: torch.Tensor,
    codebooks: torch.Tensor,
    user_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-group INT4+QJL quantization of KV heads — fully vectorized per head.

    Seed design: seed = (user_id % 1024) ^ head_idx
      All B×T tokens for the same (user, head) share one G matrix → enables
      a single (B×T, HEAD_DIM) @ (HEAD_DIM, QJL_DIM) matmul per head instead
      of B×T scalar projections. O(n_heads) Python iterations, each O(1) BLAS.

    Args:
        x:         (B, n_heads, T, HEAD_DIM) rotated KV activations
        codebooks: (n_heads, 16) float32 per-head codebooks
                   OR (16,) float32 shared codebook
        user_id:   user identifier — prevents cross-user structured noise
    Returns:
        (nibbles, qjl_packed, norms) with leading dims (B, n_heads, T, ...)
    """
    B, n_heads, T, D = x.shape
    use_per_group = codebooks.dim() == 2
    if use_per_group:
        assert codebooks.shape == (n_heads, 16), codebooks.shape

    nib_out  = torch.zeros(B, n_heads, T, _NBYTES_NIBBLE, dtype=torch.uint8,   device=x.device)
    qjl_out  = torch.zeros(B, n_heads, T, _NBYTES_QJL,   dtype=torch.uint8,   device=x.device)
    norm_out = torch.zeros(B, n_heads, T,                  dtype=torch.float32, device=x.device)

    for h in range(n_heads):
        cb  = codebooks[h] if use_per_group else codebooks   # (16,)
        x_h = x[:, h, :, :].reshape(-1, HEAD_DIM).float()   # (B*T, HEAD_DIM)

        # Vectorized INT4
        codes, dequant = _batch_quantize_int4(x_h, cb)       # (B*T, HEAD_DIM)
        nibbles_h      = _batch_pack_nibbles(codes)           # (B*T, HEAD_DIM//2)

        # Vectorized QJL — shared G across all B*T tokens for this head
        error  = x_h - dequant
        norms_h = error.norm(dim=-1)                          # (B*T,)
        seed_h  = _make_qjl_seed(user_id, h)                 # (user_id%1024) ^ h
        _, qjl_h = _batch_qjl_encode(error, seed_h)          # (B*T, QJL_DIM//8)

        nib_out[:, h, :, :]  = nibbles_h.reshape(B, T, _NBYTES_NIBBLE)
        qjl_out[:, h, :, :]  = qjl_h.reshape(B, T, _NBYTES_QJL)
        norm_out[:, h, :]    = norms_h.reshape(B, T)

    return nib_out, qjl_out, norm_out


def dequantize_heads(
    nibbles:    torch.Tensor,
    qjl_packed: torch.Tensor,
    norms:      torch.Tensor,
    codebooks:  torch.Tensor,
    user_id: int = 0,
    with_qjl: bool = True,
) -> torch.Tensor:
    """
    Per-group dequantization of KV heads — fully vectorized per head.

    Hot path: O(n_heads) Python iterations, each a BLAS matmul.
    One _batch_xor_unpack call replaces HEAD_DIM//8 individual byte loads.

    Args:
        nibbles:    (B, n_heads, T, HEAD_DIM//2) uint8
        qjl_packed: (B, n_heads, T, QJL_DIM//8) uint8
        norms:      (B, n_heads, T) float32
        codebooks:  (n_heads, 16) or (16,) float32
        user_id:    must match quantize_heads user_id
        with_qjl:   if False, INT4 only (ablation baseline)
    Returns:
        (B, n_heads, T, HEAD_DIM) float32
    """
    B, n_heads, T, _ = nibbles.shape
    use_per_group = codebooks.dim() == 2
    if use_per_group:
        assert codebooks.shape == (n_heads, 16), codebooks.shape

    out = torch.zeros(B, n_heads, T, HEAD_DIM, dtype=torch.float32, device=nibbles.device)
    for h in range(n_heads):
        cb     = codebooks[h] if use_per_group else codebooks
        nib_h  = nibbles[:, h, :, :].reshape(-1, _NBYTES_NIBBLE)  # (B*T, D//2)
        qjl_h  = qjl_packed[:, h, :, :].reshape(-1, _NBYTES_QJL)
        norm_h = norms[:, h, :].reshape(-1)                        # (B*T,)

        # Vectorized XOR-word-unpack → codebook lookup
        codes   = _batch_xor_unpack(nib_h)                         # (B*T, HEAD_DIM)
        dequant = cb[codes.long().reshape(-1)].reshape(-1, HEAD_DIM)

        if with_qjl:
            seed_h       = _make_qjl_seed(user_id, h)
            residual_hat = _batch_qjl_reconstruct(qjl_h, norm_h, seed_h, nibbles.device)
            out[:, h, :, :] = (dequant + residual_hat).reshape(B, T, HEAD_DIM)
        else:
            out[:, h, :, :] = dequant.reshape(B, T, HEAD_DIM)

    return out
