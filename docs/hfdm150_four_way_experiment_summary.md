# H-FDM 150-Seed Model Four-Way MPPI Experiment

Date: 2026-05-24

## Setup

- Model: `results/hfdm_training/geomapping_data1_150_h25_20260523_221211/export_cuda_trace`
- Trial output: `results/mppi_tuning/20260524_112625_hfdm150_four_way_5seeds`
- Goal: `(18.0, 5.0)`, tolerance `0.5 m`, timeout `180 s`
- Seeds: `424242`, `424243`, `424244`, `424245`, `424246`
- Overlay figure: `results/mppi_tuning/20260524_112625_hfdm150_four_way_5seeds/trajectory_overlay_four_way_all_5seeds.png`
- Trusted metrics source: `results/mppi_tuning/20260524_112625_hfdm150_four_way_5seeds/summary.json`

The four compared conditions were:

1. `frontend_no_fdm`: frontend + nominal MPPI
2. `frontend_hfdm`: frontend + H-FDM MPPI
3. `no_frontend_no_fdm`: no frontend + nominal MPPI
4. `no_frontend_hfdm`: no frontend + H-FDM MPPI

## Important Runtime Note

The original trained export at
`results/hfdm_training/geomapping_data1_150_h25_20260523_221211/export/fdm_ts.pt`
was CPU-traced. In CUDA controller runtime it failed because the TorchScript GRU
hidden state allocation stayed on CPU while inputs were on `cuda:0`.

A CUDA-traced export was generated and used for this experiment:

`results/hfdm_training/geomapping_data1_150_h25_20260523_221211/export_cuda_trace`

This export was verified with CUDA inference before the navigation sweep.

## Aggregate Results

| Condition | Reached | Mean arrival over successes | Mean path length | Mean final distance | Mean MPPI compute |
|---|---:|---:|---:|---:|---:|
| frontend + nominal MPPI | 5/5 | 52.82 s | 21.16 m | 0.391 m | 15.94 ms |
| frontend + H-FDM MPPI | 5/5 | 60.44 s | 31.56 m | 0.392 m | 80.95 ms |
| no frontend + nominal MPPI | 1/5 | 23.40 s | 15.41 m | 5.574 m | 8.32 ms |
| no frontend + H-FDM MPPI | 5/5 | 46.52 s | 22.40 m | 0.373 m | 49.51 ms |

## Interpretation

H-FDM is not simply ignoring the frontend because it has "learned enough". The
learned rollout gives MPPI a different motion consequence and risk model, but it
still optimizes inside the same MPPI objective. When the frontend is present,
the path prior still changes the action distribution and trajectory shape.

The key result is that removing the path frontend breaks nominal MPPI on most
seeds: `no_frontend_no_fdm` reached only `1/5`. In contrast, `no_frontend_hfdm`
reached `5/5`, which shows that the learned rollout provides useful goal-seeking
and risk-aware behavior even without the frontend path.

However, H-FDM with the frontend is not yet strictly better than nominal
frontend MPPI on this small sweep. It reached all seeds, but it was slower on
average and produced longer trajectories. This suggests the current H-FDM cost
profile still needs tuning: the model helps robustness, especially without a
frontend, but the combined learned rollout plus retained nominal costs can still
produce circuitous behavior.

## Practical Conclusion

- Do not remove the frontend as a default system component yet.
- H-FDM can run without the frontend and is much more robust than nominal MPPI in
  that mode on these five seeds.
- The strongest next experiment is not "frontend or no frontend" only; it is
  H-FDM cost-profile tuning with the frontend retained, especially reducing
  nominal-era penalties that fight learned rollout while keeping enough regularity
  for stable control.

## Follow-Up Targets

- Patch the high-level FDM export path so CUDA runtime exports do not require a
  manual CUDA trace workaround.
- Run a larger seed sweep after tuning, because five seeds are useful for
  diagnosis but not enough for final claims.
- Compare H-FDM frontend vs no-frontend on path length, arrival time, and failure
  modes after reducing conservative nominal penalties.
