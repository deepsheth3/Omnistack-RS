"""
OmniStack-RS — MLPerf LoadGen Server Scenario (OmniStack Open benchmark)

Drives the fused INT4+QJL attention path using the official mlcommons_loadgen
library. This is **not** a MLPerf DLRMv3 closed-division submission; it is an
Open-division style measurement of the OmniStack-RS kernel using LoadGen
scheduling, logging, and Server-scenario traffic (Poisson / target QPS).

Pipeline:
  Stages 1–2: same transcoder + bake as run_criteo_benchmark.py
  Stage 3:    LoadGen IssueQuery → omni_attn → QuerySamplesComplete

Usage:
  python scripts/run_criteo_loadgen.py --synthetic --fast
  python scripts/run_criteo_loadgen.py --criteo-path data/day_23_sample.tsv
  python scripts/run_criteo_loadgen.py --help
"""

from __future__ import annotations

import argparse
import array
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Optional: official LoadGen
try:
    import mlperf_loadgen as lg
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "mlperf_loadgen is required. Install with: pip install mlcommons-loadgen"
    ) from e

from scripts.run_criteo_benchmark import (  # noqa: E402
    BATCH_SIZE,
    N_KV_HEADS,
    N_Q_HEADS,
    HIDDEN_DIM,
    LORA_RANK,
    N_LORAS,
    SEQ_LEN,
    _build_lora_params,
    _build_synthetic_users,
    _time_kernel_call,
    bake_kv_cache,
    load_criteo_sequences,
)
from omnistack_rs.kernels.fused_attention import make_g_matrix
from omnistack_rs.kernels.hadamard import HEAD_DIM


@dataclass
class LoadGenConfig:
    criteo_path: Optional[str] = None
    synthetic: bool = False
    n_users: int = 256
    max_rows: int = 5_000_000
    bake_cache: str = "criteo_baked.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # LoadGen
    log_outdir: str = "mlperf_logs"
    server_target_qps: float = 1000.0
    server_target_latency_ns: int = 100_000_000  # 100 ms MLPerf-style bound
    min_query_count: int = 270_336
    min_duration_ms: int = 60_000
    max_duration_ms: int = 0
    mode: str = "performance"  # performance | accuracy
    fast: bool = False
    enable_trace: bool = False
    # One logical “user” = one QSL sample; batch = all samples in the LoadGen query
    batch_size_cap: int = BATCH_SIZE  # cap batch to stay within memory


class OmniStackSUT:
    """System Under Test: runs omni_attn for each LoadGen query batch."""

    def __init__(
        self,
        nibbles: torch.Tensor,
        qjl: torch.Tensor,
        norms: torch.Tensor,
        codebooks: torch.Tensor,
        device: torch.device,
        seed: int,
        batch_size_cap: int = BATCH_SIZE,
    ) -> None:
        self.batch_size_cap = int(batch_size_cap)
        self.nibbles = nibbles.to(device)
        self.qjl = qjl.to(device)
        self.norms = norms.to(device)
        self.codebooks = codebooks.to(device)
        self.device = device
        self.seed = seed
        self.N = nibbles.shape[0]
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

        self.g_matrix = make_g_matrix(N_KV_HEADS, seed=seed, device=device)
        self.lora_a, self.lora_b = _build_lora_params(
            N_LORAS, LORA_RANK, HIDDEN_DIM, N_Q_HEADS, device, seed=seed
        )
        self.torch_rng = torch.Generator(device=device)
        self.torch_rng.manual_seed(seed)

    def issue_query(self, query_samples: List) -> None:
        """
        Synchronous path: run one forward per LoadGen query and complete immediately.
        Each LoadGen sample id maps to a user row index in [0, N).
        """
        if not query_samples:
            return

        B = min(len(query_samples), self.batch_size_cap)  # align with server batching
        samples = query_samples[:B]

        # User indices = QSL sample indices
        user_idxs = []
        for s in samples:
            u = int(s.index) % self.N
            user_idxs.append(u)
        idx = torch.tensor(user_idxs, device=self.device, dtype=torch.long)

        # Reproducible per-query noise: seed from first sample id
        sid = int(samples[0].id)
        self.torch_rng.manual_seed((self.seed ^ sid) & 0x7FFFFFFF)

        q = torch.randn(
            B, N_Q_HEADS, 1, HEAD_DIM, generator=self.torch_rng, device=self.device
        )
        x = torch.randn(B, 1, HIDDEN_DIM, generator=self.torch_rng, device=self.device)
        lora_idx = torch.randint(
            -1, N_LORAS, (B,), generator=self.torch_rng, device=self.device
        ).to(torch.int32)

        nib_b = self.nibbles[idx]
        qjl_b = self.qjl[idx]
        norm_b = self.norms[idx]

        out, _ = _time_kernel_call(
            q,
            nib_b,
            qjl_b,
            norm_b,
            nib_b,
            self.codebooks,
            self.g_matrix,
            self.lora_a,
            self.lora_b,
            lora_idx,
            x,
            self.scale,
            self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        responses: List = []
        if self.accuracy_mode:
            if not hasattr(self, "_acc_buffers"):
                self._acc_buffers = []  # keep arrays alive until QuerySamplesComplete returns
            # One float32 tag per sample for mlperf_log_accuracy.json
            per = max(1, out.numel() // max(B, 1))
            for i, s in enumerate(samples):
                off = (i * per) % out.numel()
                v = float(out.reshape(-1)[off].detach().float().cpu().item())
                a = array.array("f", [v])
                self._acc_buffers.append(a)
                ptr, n = a.buffer_info()
                responses.append(lg.QuerySampleResponse(s.id, ptr, n * a.itemsize))
        else:
            for s in samples:
                responses.append(lg.QuerySampleResponse(s.id, 0, 0))

        lg.QuerySamplesComplete(responses)
        if self.accuracy_mode and hasattr(self, "_acc_buffers"):
            self._acc_buffers.clear()

    def flush_queries(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    accuracy_mode: bool = False


def _load_stages_1_2(cfg: LoadGenConfig):
    if cfg.synthetic:
        print("[Stage 1] Synthetic sequences...")
        kv = _build_synthetic_users(cfg.n_users, SEQ_LEN, N_KV_HEADS, cfg.seed)
    else:
        kv = load_criteo_sequences(
            str(cfg.criteo_path), cfg.n_users, SEQ_LEN, N_KV_HEADS, cfg.max_rows, cfg.seed
        )
    print(f"  KV shape: {tuple(kv.shape)}")

    nibbles, qjl, norms, codebooks, _stats = bake_kv_cache(
        kv, cfg.bake_cache, force=False, seed=cfg.seed
    )
    return nibbles, qjl, norms, codebooks


def parse_args() -> LoadGenConfig:
    p = argparse.ArgumentParser(description="OmniStack-RS + MLPerf LoadGen (Server)")
    p.add_argument("--criteo-path", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-users", type=int, default=256)
    p.add_argument("--max-rows", type=int, default=5_000_000)
    p.add_argument("--bake-cache", type=str, default="criteo_baked.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-outdir", type=str, default="mlperf_logs")
    p.add_argument("--server-target-qps", type=float, default=1634.0)
    p.add_argument("--min-query-count", type=int, default=270_336)
    p.add_argument("--min-duration-ms", type=int, default=60_000)
    p.add_argument("--max-duration-ms", type=int, default=0)
    p.add_argument("--fast", action="store_true", help="Short run for local smoke test")
    p.add_argument(
        "--enable-trace",
        action="store_true",
        help="Write mlperf_log_trace.json (large). Default: off for smaller logs.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="performance",
        choices=["performance", "accuracy"],
    )
    args = p.parse_args()
    if not args.synthetic and not args.criteo_path:
        p.error("Either --synthetic or --criteo-path is required")
    if args.fast:
        # Enough queries for early-stopping stats; keep duration short on CPU
        args.min_query_count = 500
        args.min_duration_ms = 5_000
        # High target QPS on CPU / Triton interpret fails latency SLO; use modest rate for smoke
        if args.server_target_qps == 1000.0:
            args.server_target_qps = 20.0
    cfg = LoadGenConfig(
        criteo_path=args.criteo_path,
        synthetic=args.synthetic,
        n_users=args.n_users,
        max_rows=args.max_rows,
        bake_cache=args.bake_cache,
        device=args.device,
        seed=args.seed,
        log_outdir=args.log_outdir,
        server_target_qps=args.server_target_qps,
        min_query_count=args.min_query_count,
        min_duration_ms=args.min_duration_ms,
        max_duration_ms=args.max_duration_ms,
        mode=args.mode,
        fast=args.fast,
        enable_trace=args.enable_trace,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    print("=" * 72, flush=True)
    print("OmniStack-RS + MLPerf LoadGen — Server scenario")
    print(f"  Device: {cfg.device}  |  n_users: {cfg.n_users}  |  fast: {cfg.fast}")
    print(f"  Target QPS: {cfg.server_target_qps}  |  min_query_count: {cfg.min_query_count}")
    print("=" * 72)

    nibbles, qjl, norms, codebooks = _load_stages_1_2(cfg)
    n_total = int(nibbles.shape[0])

    device = torch.device(cfg.device)
    sut = OmniStackSUT(
        nibbles, qjl, norms, codebooks, device, cfg.seed, batch_size_cap=cfg.batch_size_cap
    )
    sut.accuracy_mode = cfg.mode == "accuracy"

    def load_samples_to_ram(slist: List[int]) -> None:  # noqa: ARG001
        return

    def unload_samples_from_ram(slist: List[int]) -> None:  # noqa: ARG001
        return

    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.Server
    if cfg.mode == "performance":
        settings.mode = lg.TestMode.PerformanceOnly
    else:
        settings.mode = lg.TestMode.AccuracyOnly
    settings.server_target_qps = float(cfg.server_target_qps)
    settings.server_target_latency_ns = int(cfg.server_target_latency_ns)
    settings.min_query_count = int(cfg.min_query_count)
    settings.min_duration_ms = int(cfg.min_duration_ms)
    if cfg.max_duration_ms and cfg.max_duration_ms > 0:
        settings.max_duration_ms = int(cfg.max_duration_ms)

    log = lg.LogSettings()
    out = Path(cfg.log_outdir)
    out.mkdir(parents=True, exist_ok=True)
    log.log_output = lg.LogOutputSettings()
    log.log_output.outdir = str(out.resolve())
    # Summary is always written to mlperf_log_summary.txt; avoid interleaving with our prints
    log.log_output.copy_summary_to_stdout = False
    if hasattr(log, "enable_trace"):
        log.enable_trace = bool(cfg.enable_trace)

    print(f"\n[LoadGen] Output directory: {out.resolve()}\n")
    print("[LoadGen] Starting test...")

    qsl = lg.ConstructQSL(
        n_total, n_total, load_samples_to_ram, unload_samples_from_ram
    )
    sut_h = lg.ConstructSUT(sut.issue_query, sut.flush_queries)

    lg.StartTestWithLogSettings(sut_h, qsl, settings, log)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut_h)

    print(f"\n[LoadGen] Done. Logs in: {out.resolve()}")


if __name__ == "__main__":
    main()
