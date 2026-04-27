# OmniStack-RS — code snapshot pointer

**Canonical source:** [https://github.com/deepsheth3/Omnistack-RS](https://github.com/deepsheth3/Omnistack-RS)

This folder exists to satisfy a submission-style tree layout. The real repository contains:

- `omnistack_rs/kernels/fused_attention.py` — Triton fused attention
- `scripts/run_criteo_benchmark.py` — Criteo → bake → hand-rolled Server timing
- `scripts/run_criteo_loadgen.py` — Criteo → bake → **MLPerf LoadGen** Server
- `benchmark_proofs/` — A10 scorecards, Nsight artifacts

**Open division:** custom INT4+QJL codec + GQA + Multi-LoRA, not the closed **DLRMv3** graph.
