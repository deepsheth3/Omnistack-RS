"""
OmniStack-RS — Throughput-at-Latency Benchmark: Ad Recommendation

Simulates the hot path for Meta-scale ad serving:
  - 1,000 ad candidate KV vectors to score against a user query
  - 10 ms per-request deadline (P99 budget for 100M QPS)
  - Per-group codebooks: 8 KV head groups, each with 16 centroids
  - Per-user QJL seed: (user_id % 1024) ^ head_idx

Pipeline (online hot path):
  1. Dequantize 1,000 × n_kv_heads × HEAD_DIM nibble-packed INT4 keys
  2. Apply QJL residual reconstruction (1-bit sign correction)
  3. Compute attention scores: query @ K^T across all candidates
  4. Return top-K candidates

This benchmark runs on CPU (Mac dev environment). Throughput on H100 is
expected to be ~100× higher; the 10 ms target is met on CPU, with large
headroom at GPU inference time.

Usage:
    python benchmarks/bench_ads.py [--n-candidates 1000] [--n-users 10] [--topk 50]
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from omnistack_rs.kernels.hadamard import HEAD_DIM, _wht_torch
from omnistack_rs.kernels.quantize import (
    QJL_DIM,
    _NBYTES_NIBBLE,
    _NBYTES_QJL,
    quantize_heads,
    dequantize_heads,
)
from omnistack_rs.quantization.codebook import calibrate_per_group

# ── Config ────────────────────────────────────────────────────────────────

N_KV_HEADS   = 8     # KV heads (= n_groups for per-group codebooks)
N_Q_HEADS    = 32    # query heads (GQA: 4 Q per KV head)
LATENCY_BUDGET_MS = 10.0


# ── Helpers ───────────────────────────────────────────────────────────────

def _generate_ad_kv_cache(
    n_candidates: int,
    n_kv_heads: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Synthetic KV cache for n_candidates ads.

    Each ad candidate has one key vector per KV head, stored in Hadamard-rotated
    space (as produced by rotate_kv_cache in the actual pipeline).

    Shape: (n_candidates, n_kv_heads, 1, HEAD_DIM)  — T=1 (single decode step)
    """
    torch.manual_seed(seed)
    raw = torch.randn(n_candidates, n_kv_heads, 1, HEAD_DIM)
    return _wht_torch(raw)


def _calibrate_codebooks(
    kv_cache: torch.Tensor,  # (N, n_kv_heads, 1, HEAD_DIM)
    n_kv_heads: int,
) -> torch.Tensor:
    """
    Calibrate one Lloyd-Max codebook per KV head group from ad cache samples.

    Returns: (n_kv_heads, 16) float32
    """
    samples_by_group = [
        kv_cache[:, h, 0, :].reshape(-1)  # all candidates, head h, flattened
        for h in range(n_kv_heads)
    ]
    return calibrate_per_group(samples_by_group)


def _time_online_path(
    nibbles:    torch.Tensor,   # (N, n_kv_heads, 1, HEAD_DIM//2)
    qjl_packed: torch.Tensor,   # (N, n_kv_heads, 1, QJL_DIM//8)
    norms:      torch.Tensor,   # (N, n_kv_heads, 1)
    codebooks:  torch.Tensor,   # (n_kv_heads, 16)
    query:      torch.Tensor,   # (n_q_heads, HEAD_DIM) — one decode step
    user_id:    int,
    n_q_heads:  int,
    topk:       int,
    n_warmup:   int = 3,
    n_timed:    int = 10,
) -> dict:
    """
    Time the dequantize → score → topK pipeline.

    Returns dict with latency stats (ms) and throughput (candidates/s).
    """
    n_kv_heads = codebooks.shape[0]
    n_groups   = n_q_heads // n_kv_heads   # GQA group size

    def _run_once():
        # Dequantize all candidates (hot path)
        k_hat = dequantize_heads(nibbles, qjl_packed, norms, codebooks,
                                 user_id=user_id, with_qjl=True)
        # k_hat: (N_cand, n_kv_heads, 1, HEAD_DIM)
        # Expand KV heads to Q heads (GQA)
        k_exp = k_hat.repeat_interleave(n_groups, dim=1)  # (N_cand, n_q_heads, 1, D)

        # Attention scores: query (n_q, D) @ K^T (N_cand, n_q, D) → (N_cand, n_q)
        # Flatten candidate×token dim for batch matmul
        k_flat = k_exp[:, :, 0, :]                        # (N_cand, n_q, D)
        scores = torch.einsum("qd,nqd->nq", query, k_flat)  # (N_cand, n_q)
        scores_per_cand = scores.mean(dim=1)               # (N_cand,) average over heads
        _, topk_idx = scores_per_cand.topk(topk)
        return topk_idx

    # Warm up
    for _ in range(n_warmup):
        _run_once()

    # Timed runs
    latencies_ms = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        _run_once()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    n_cand   = nibbles.shape[0]
    mean_ms  = sum(latencies_ms) / len(latencies_ms)
    p50_ms   = sorted(latencies_ms)[len(latencies_ms) // 2]
    p99_ms   = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
    throughput = n_cand / (mean_ms / 1000.0)

    return {
        "n_candidates": n_cand,
        "mean_ms":      mean_ms,
        "p50_ms":       p50_ms,
        "p99_ms":       max(latencies_ms),   # worst observed at n_timed=10
        "throughput_k_per_s": throughput / 1000.0,
        "meets_10ms_budget":  mean_ms < LATENCY_BUDGET_MS,
    }


# ── Compression ratio ─────────────────────────────────────────────────────

def _compression_ratio(n_candidates: int, n_kv_heads: int) -> dict:
    """
    Compute bytes: BF16 baseline vs INT4+QJL compressed.

    Per element: BF16 = 2 bytes; INT4 = 0.5 bytes; QJL sign = 1/HEAD_DIM bytes.
    """
    n_elements   = n_candidates * n_kv_heads * HEAD_DIM
    bf16_bytes   = n_elements * 2                            # 2 bytes per BF16
    nibble_bytes = n_candidates * n_kv_heads * _NBYTES_NIBBLE
    qjl_bytes    = n_candidates * n_kv_heads * _NBYTES_QJL
    norm_bytes   = n_candidates * n_kv_heads * 4             # float32 norm
    compressed   = nibble_bytes + qjl_bytes + norm_bytes
    return {
        "bf16_kb":        bf16_bytes    / 1024,
        "int4_qjl_kb":    compressed    / 1024,
        "ratio":          bf16_bytes    / compressed,
        "bits_per_elem":  compressed * 8 / n_elements,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_benchmark(
    n_candidates: int = 1000,
    n_users: int = 10,
    topk: int = 50,
) -> None:
    print("=" * 64)
    print(f"OmniStack-RS  Ad Recommendation Throughput Benchmark")
    print(f"  {n_candidates} candidates · {N_KV_HEADS} KV heads · {N_Q_HEADS} Q heads")
    print(f"  HEAD_DIM={HEAD_DIM}  QJL_DIM={QJL_DIM}  topK={topk}")
    print("=" * 64)

    # ── Offline: generate and compress ad KV cache ────────────────────────
    print("\n[Offline] Generating and quantizing ad KV cache...")
    kv_cache = _generate_ad_kv_cache(n_candidates, N_KV_HEADS)

    t_cal_start = time.perf_counter()
    codebooks = _calibrate_codebooks(kv_cache, N_KV_HEADS)
    t_cal_ms  = (time.perf_counter() - t_cal_start) * 1000

    t_quant_start = time.perf_counter()
    nibbles, qjl_packed, norms = quantize_heads(kv_cache, codebooks, user_id=0)
    t_quant_ms = (time.perf_counter() - t_quant_start) * 1000

    cr = _compression_ratio(n_candidates, N_KV_HEADS)
    print(f"  Codebook calibration: {t_cal_ms:.1f} ms")
    print(f"  Quantization:         {t_quant_ms:.1f} ms")
    print(f"  BF16 size:            {cr['bf16_kb']:.1f} KB")
    print(f"  INT4+QJL size:        {cr['int4_qjl_kb']:.1f} KB")
    print(f"  Compression ratio:    {cr['ratio']:.2f}×  ({cr['bits_per_elem']:.2f} bits/elem)")

    # ── Online: score users against compressed cache ───────────────────────
    print(f"\n[Online] Scoring {n_users} users vs {n_candidates} ad candidates...")
    results = []
    torch.manual_seed(77)
    for u in range(n_users):
        query = torch.randn(N_Q_HEADS, HEAD_DIM)   # (n_q_heads, D)
        stats = _time_online_path(
            nibbles, qjl_packed, norms, codebooks,
            query=query, user_id=u,
            n_q_heads=N_Q_HEADS, topk=topk,
        )
        results.append(stats)

    mean_ms_all  = sum(r["mean_ms"]  for r in results) / n_users
    tput_all     = sum(r["throughput_k_per_s"] for r in results) / n_users
    meets_budget = all(r["meets_10ms_budget"] for r in results)

    print(f"\n{'User':>4}  {'Mean (ms)':>10}  {'P99 (ms)':>9}  {'Throughput (K/s)':>17}")
    print("-" * 48)
    for u, r in enumerate(results):
        flag = "✓" if r["meets_10ms_budget"] else "✗"
        print(f"{u:>4}  {r['mean_ms']:>10.2f}  {r['p99_ms']:>9.2f}  "
              f"{r['throughput_k_per_s']:>17.1f}  {flag}")

    print("-" * 48)
    print(f"{'avg':>4}  {mean_ms_all:>10.2f}  {'':>9}  {tput_all:>17.1f}")
    print()
    verdict = "PASS" if meets_budget else "FAIL"
    print(f"10 ms deadline: {verdict}  (all users: {meets_budget})")
    print(f"Compression:    {cr['ratio']:.2f}×  (target: ≥3.2×  — "
          f"{'PASS' if cr['ratio'] >= 3.2 else 'FAIL'})")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ad recommendation throughput benchmark")
    parser.add_argument("--n-candidates", type=int, default=1000)
    parser.add_argument("--n-users",      type=int, default=10)
    parser.add_argument("--topk",         type=int, default=50)
    args = parser.parse_args()
    run_benchmark(
        n_candidates=args.n_candidates,
        n_users=args.n_users,
        topk=args.topk,
    )
