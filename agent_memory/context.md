# 项目上下文

## 项目目标

- 当前有效目标：以旧 60-seed/600-episode H-FDM H25 权重作为最终权重候选，使用 `nf_green_mild_noise_jerk` 作为最终 MPPI 控制 profile，准备 no-frontend H-FDM 的最终复验和论文图表。

## 关键背景

- 相关仓库、模块或文件：`src/mppi_controller/configs/mujoco_rviz_goal_hfdm_h25.yaml`、`src/ausim_geomapping_adapter/launch/ausim_scout_mppi_hfdm_no_frontend.launch.py`、`tools/run_hfdm_model_seed_sweep.py`、`tools/run_hfdm_no_frontend_candidate_sweep.py`、`tools/geomapping_nav_trial.py`、`tools/analyze_hfdm_sweep.py`、`results/mppi_tuning/20260524_hfdm60_two_way_5seeds`、`results/mppi_tuning/20260524_hfdm_no_frontend_seed424245_fix/profiles/nf_green_mild_noise_jerk.yaml`。
- 重要约束：H-FDM 最终实验不需要跑 frontend 版本；只做 no-frontend/direct navigation；不改 MPPI 核心逻辑，不重训 H-FDM。
- 已确认假设：轨迹“绕来绕去”按路径波浪、横向摆动、角速度高频换向和终点捕获失败衡量，而不是只按是否到达衡量。

## 当前约定

- 沟通与协作约定：默认中文沟通；代码和配置标识保持仓库风格。
- 架构或实现约定：H-FDM 使用 `fdm.mode=high_level_fdm`；调参输出放在 `results/mppi_tuning` 下，不直接覆盖源 profile，除非最终结果明确优于基线。
- 验证约定：先用代表 seeds 快筛候选，再对最优候选跑完整 5 seeds；每次结论以 `native_run/summary.json` 和 `tools/analyze_hfdm_sweep.py` 指标为准。

## 最近结论

- 最终计划不再把 `nf_yaw_light_filter` 作为主 profile；当前 source H-FDM profile `src/mppi_controller/configs/mujoco_rviz_goal_hfdm_h25.yaml` 已同步为 `nf_green_mild_noise_jerk` 控制参数。
- `nf_green_mild_noise_jerk` 的关键控制参数：`std_normal=[0.50,0.19,0.25]`，`smooth_weight=0.028`，`lateral_weight=0.04`，`yaw_rate_weight=0.04`，`jerk_weight=0.012`，`update_smoothing_alpha=[0.0,0.03,0.04]`，`command_filter.alpha=0.02`，deadband `0.008`。
- 该修复 profile 已在 seed `424245` 上验证：fresh run `results/mppi_tuning/20260524_214107_hfdm150_seed424245_nf_green_mild_noise_jerk` 到达，arrival `43.6s`、path `21.48m`、final distance `0.387m`，未复现 tight-noise 的 `89.7s/40.52m` 大回环。
- 当前 source H-FDM profile 保留 60-seed 权重 `/home/mexxiie/prj/high_level_fdm/runs/geomapping_data1_h25/export`，并默认 no-frontend：`external_path.enabled=false`、`global_path.enabled=false`、`path_tracking_weight=0`、`path_progress_weight=0`。后续仍需跑 no-frontend H-FDM final sweep；H-FDM frontend 不作为必要条件。
