# MLPerf-style Open benchmark layout (OmniStack-RS)

This directory mirrors a **submission-style** folder tree for documentation and for copying **MLPerf LoadGen** outputs after a run.

**Important:** This is **not** a closed-division **DLRMv3** MLPerf submission. It is an **Open** custom benchmark of the OmniStack-RS fused attention + INT4+QJV codec, driven by the official **mlcommons-loadgen** Server scenario.

| Path | Purpose |
|------|--------|
| [systems/A10_omnistack.json](systems/A10_omnistack.json) | System description (A10, cloud) |
| [model_mapping.json](model_mapping.json) | Open custom model name mapping |
| [measurements/.../measurements.json](measurements/A10_omnistack/omnistack-rs/server/measurements.json) | Hardware, software, technique |
| [measurements/.../user.conf](measurements/A10_omnistack/omnistack-rs/server/user.conf) | LoadGen overrides (match `run_criteo_loadgen.py` defaults) |
| [results/.../performance/run_1/README.md](results/A10_omnistack/omnistack-rs/server/performance/run_1/README.md) | Where to copy `mlperf_log_*.txt` after a performance run |
| [code/omnistack-rs/README.md](code/omnistack-rs/README.md) | Pointer to the GitHub repo |

## Run LoadGen

```bash
pip install "mlcommons-loadgen>=5"   # or: pip install -e ".[mlperf]"
python scripts/run_criteo_loadgen.py --synthetic --log-outdir ./mlperf_logs
```

Use `--fast` for a short local smoke test. Use `--help` for all options.

## Official submission checker

Passing `submission_checker.py` for a **custom Open** benchmark may require additional fields and exact layout for a given MLPerf round. Use this tree as a starting point and align with the [MLPerf Inference submission docs](https://docs.mlcommons.org/inference/submission/) for the round you target.
