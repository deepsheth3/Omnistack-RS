# Performance run (LoadGen)

After running:

```bash
pip install mlcommons-loadgen
python scripts/run_criteo_loadgen.py --synthetic --log-outdir ./mlperf_logs
# or: --criteo-path data/day_23_sample.tsv
```

Copy into this directory:

- `mlperf_log_summary.txt`
- `mlperf_log_detail.txt`
- `mlperf_log_trace.json` (only if you passed `--enable-trace`; large)

These are produced by **MLPerf LoadGen** (`mlcommons-loadgen`) in Server scenario.

**Note:** This is an **Open-division** measurement of the OmniStack-RS kernel, not a closed-division **DLRMv3** official task submission.
