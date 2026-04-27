# Accuracy run (LoadGen)

Run with:

```bash
python scripts/run_criteo_loadgen.py --synthetic --mode accuracy --log-outdir ./mlperf_acc_logs
```

Copy into this directory:

- `mlperf_log_summary.txt`
- `mlperf_log_detail.txt`
- `mlperf_log_accuracy.json`
- `accuracy.txt` (optional: your own text summary of parity vs reference)

OmniStack-RS accuracy mode returns one float per sample in `mlperf_log_accuracy.json` for log sanity; for strict MLPerf task accuracy, use the task’s official eval scripts (N/A for this custom Open benchmark).
