Measured JSON goes here and is committed.

`{ceilings,stages,sweep,stamp}_{T4|A100}.json`

A100 diagnostic: `diag_rows_A100.json` (v1 at 11008×4096), `diag_wpr{2,4}_A100.json` (v1 with 2/4 warps per row).

T4 `stages_T4.json` is currently 4096×11008 (the 4096×4096 T4 stage table from the first Modal run was overwritten; those medians are kept in RESULTS.md). Hit rates were recomputed against the read-only STREAM roof without retiming the GEMV kernels.

`ceilings_*.json` reports `hbm_copy` (kind `d2d_copy`, factor 2), `hbm_read` (kind `stream_read_sum`, factor 1), and `hbm_write` (kind `stream_write_fill`, factor 1). GEMV `% HBM` uses `hbm_read`.

Each file’s `meta` includes `gpu_type`, `driver_version`, and `nvidia_smi`.
