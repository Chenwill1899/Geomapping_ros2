# 当前任务进度

## 当前任务

- 任务：调优当前 150-seed H-FDM no-frontend 约束/代价系数。
- 成功标准：完整 5 seeds 保持 `5/5` 到达；平均 arrival 尽量不高于基线 `46.52s`，至少不明显恶化；`control_wz_sign_switches`、`total_heading_change`、`mean_abs_vy` 或 jerk 至少有一组关键摆动指标明显下降。
- 范围边界：只生成/验证 no-frontend H-FDM profile 候选；不重训、不改 MPPI 核心逻辑、不切换到 60 权重作为主方案。
- 停止条件：候选连续 3 次无法同时保持到达和时间，或运行环境失败无法恢复。

## 已完成

- 已补齐并读取 `agent_memory/` 模板文件。
- 已读取 `.omx/project-memory.json`、H-FDM/nominal/no-frontend profile、four-way sweep 脚本和现有 summary。
- 已新增 `tools/analyze_hfdm_sweep.py` 和对应 pytest，生成现有 5-seed 离线诊断报告。
- 已生成 no-frontend H-FDM 稳定性 tuned v1 profile 候选。
- 已跑 seed `424242` 的 tuned v1 冒烟导航验证。
- 已定位旧权重为 `/home/mexxiie/prj/high_level_fdm/runs/geomapping_data1_h25/export`，metadata 为 H25，数据 split 为 `480` train / `120` val episodes。
- 已新增 `tools/run_hfdm_model_seed_sweep.py`，用于指定 H-FDM model-dir 跑 frontend/no-frontend 两组并支持 resume。
- 已完成旧 60 权重的 `424242-424246` 两组 H-FDM 复跑：10/10 到达。
- 已新增 `tools/run_hfdm_no_frontend_candidate_sweep.py`，用于生成 no-frontend H-FDM 调参候选 profile，支持 `--candidates` 选择候选和 resume。
- 已完成 tuned 候选 5-seed 验证，输出目录 `results/mppi_tuning/20260524_hfdm_no_frontend_tuning_fast`。
- 已验证 `nf_yaw_light_filter` 是当前最佳折中候选：5/5 到达，平均 arrival `47.78s`，path `20.98m`，control wz switches `64.8`，heading change `48.87rad`，control jerk `0.107`。
- 已继续尝试 v2/v3/v4/v5 候选，输出目录分别为 `results/mppi_tuning/20260524_hfdm_no_frontend_tuning_v2`、`..._v3`、`..._v4`、`..._v5`。这些候选未超过 `nf_yaw_light_filter` 的 5-seed 稳健性。
- 已分析 `results/mppi_tuning/20260524_hfdm_no_frontend_tuning_v4` 中 seed `424245` 的终点附近异常：该 run 在 step `548` 到过距目标约 `0.65m`，但未进 `0.4m` controller 终止阈值，随后越过目标到 `x≈24.5,y≈7.3` 后再回头，最终 `89.7s` 到达、path `40.52m`。
- 已验证 terminal-capture 方向不可用：`nf_green_tight_noise_jerk_terminal` 只改 `final_controller.disable_when_local_costmap=false` 后，seed `424245` 动态复验超时，最终距目标约 `14.80m`，没有生成 `native_run/summary.json`；该实验候选和测试已从代码中移除。
- 已完成 seed `424245` 修复复验：`nf_green_mild_noise_jerk` 将采样噪声从 v4 tight `[0.45,0.16,0.22]` 放松到 mild `[0.50,0.19,0.25]`，fresh run `results/mppi_tuning/20260524_214107_hfdm150_seed424245_nf_green_mild_noise_jerk` 到达，controller summary 为 arrival `43.6s`、path `21.48m`、final distance `0.387m`，未复现原 v4 `89.7s/40.52m` 大回环。

## 进行中

- 准备汇报 seed `424245` 修复结果；若要推广为整组候选，还需要完整 seeds `424242-424246` 复验。

## 待处理

- 若要上线默认 no-frontend H-FDM，可将 `nf_yaw_light_filter.yaml` 整理为正式 source config；本轮未覆盖原始 source profile。
- 建议后续扩大到更多 seeds，确认 `nf_yaw_light_filter` 在动态障碍随机性下仍稳定。
- 如果要继续突破，需要考虑更大的搜索面或改 rollout/cost 结构；继续围绕当前几个标量微调收益很低。

## 验证

- 已验证：现有 H-FDM profile 中 `smooth/accel/lateral/yaw_rate/jerk/update_smoothing_alpha` 为 0，no-frontend 派生 profile 关闭 path tracking/progress。
- 已验证：`python3 -m pytest -q src/mppi_controller/test/test_hfdm_sweep_diagnostics.py` 通过；`python3 -m py_compile tools/analyze_hfdm_sweep.py src/mppi_controller/test/test_hfdm_sweep_diagnostics.py` 通过；`git diff --check` 通过；候选 profile 可被 `build_experiment_config` 解析。
- 已验证：tuned v1 seed `424242` 到达，但 arrival time `59.6s` 比 baseline `50.8s` 慢约 `17.3%`，不满足 10-15% 到达时间门槛。
- 已验证：旧 60 权重 CUDA runtime 可加载并连续推理两次；旧 60 权重复跑输出 `results/mppi_tuning/20260524_hfdm60_two_way_5seeds`。
- 已验证：旧 60 权重 frontend H-FDM `5/5` 到达，平均 arrival `40.80s`；no-frontend H-FDM `5/5` 到达，平均 arrival `55.30s`。
- 已验证：`python3 -m py_compile tools/run_hfdm_no_frontend_candidate_sweep.py tools/analyze_hfdm_sweep.py` 通过；`python3 -m pytest -q src/mppi_controller/test/test_hfdm_sweep_diagnostics.py` 通过；`git diff --check` 通过。
- 已验证：`nf_green_tight_noise_jerk` 5/5 到达但均值 path `25.04m`、arrival `51.22s`，因 seed `424245` 大绕行不应推广。
- 已验证：seed `424245` 的 `planning_goals.csv` 恒定为最终目标 `(18,5)`，障碍物列表无贴近终点障碍；异常更符合 no-frontend H-FDM 终点捕获/姿态保护不足，而不是前端路径或终点障碍导致。
- 已验证：`PYTHONPATH=src/mppi_controller python3 -m pytest -q src/mppi_controller/test/test_hfdm_sweep_diagnostics.py src/mppi_controller/test/test_mujoco_motion_policy.py` 通过，共 `10 passed`；`python3 -m py_compile tools/run_hfdm_no_frontend_candidate_sweep.py tools/analyze_hfdm_sweep.py` 通过；`git diff --check` 通过。
- 已验证：`nf_green_mild_noise_jerk` seed `424245` fresh 动态复验到达，`native_run/summary.json` 存在，arrival `43.6s`，path `21.4759m`；轨迹首次进入 1m 后最大距离仍是该首次进入点 `0.983m`，未发生越过目标后再远离的大回环。

## 下一步

- 推荐保留 `nf_yaw_light_filter` 作为当前最佳 no-frontend H-FDM tuned profile；若要继续提高视觉轨迹质量，建议先做更多 seed/场景可视化再决定是否改成本结构。
