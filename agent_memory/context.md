# 项目上下文

## 项目目标

- 当前有效目标：调优当前 150-seed H-FDM 在 no-frontend 模式下的约束/代价系数，减少绕行、横向摆动和角速度换向，同时保持 5-seed 到达率和到达时间。

## 关键背景

- 相关仓库、模块或文件：`tools/run_hfdm_four_way_seed_sweep.py`、`tools/run_hfdm_model_seed_sweep.py`、`tools/geomapping_nav_trial.py`、`tools/analyze_hfdm_sweep.py`、`src/mppi_controller/configs/mujoco_rviz_goal_hfdm_h25.yaml`、`results/mppi_tuning/20260524_112625_hfdm150_four_way_5seeds`。
- 重要约束：只调 H-FDM no-frontend profile 候选；不改 MPPI 核心逻辑，不重训 H-FDM，不切换模型作为主要优化路径。
- 已确认假设：轨迹“绕来绕去”按路径波浪、横向摆动、角速度高频换向衡量，而不是按是否到达衡量。

## 当前约定

- 沟通与协作约定：默认中文沟通；代码和配置标识保持仓库风格。
- 架构或实现约定：H-FDM 使用 `fdm.mode=high_level_fdm`；调参输出放在 `results/mppi_tuning` 下，不直接覆盖源 profile，除非最终结果明确优于基线。
- 验证约定：先用代表 seeds 快筛候选，再对最优候选跑完整 5 seeds；每次结论以 `native_run/summary.json` 和 `tools/analyze_hfdm_sweep.py` 指标为准。

## 最近结论

- 当前 150-seed no-frontend H-FDM 基线：5/5 到达，平均 arrival `46.52s`，平均 path `22.40m`，mean control wz switches `105.4`，mean total heading change `53.70rad`，mean_abs_vy `0.113`。
- 已验证 tuned 候选 `nf_yaw_light_filter`：profile 在 `results/mppi_tuning/20260524_hfdm_no_frontend_tuning_fast/profiles/nf_yaw_light_filter.yaml`，5/5 到达，平均 arrival `47.78s`，path `20.98m`，control wz switches `64.8`，heading change `48.87rad`，control jerk `0.107`。相比基线 arrival 慢约 `2.7%`，但路径和控制摆动明显下降，是当前推荐的 no-frontend H-FDM 参数候选。
- 继续围绕绿色候选尝试了 direct/silk/crisp、低滞后 filter、tight/mild sampling noise 和 soft cost 等变体。没有找到同时更短且更丝滑的 5-seed 稳健替代；`nf_green_tight_noise_jerk` 在 3-seed 快筛更快更短，但 5-seed 的 seed `424245` 出现 `40.52m` 大绕行，不能替换绿色候选。
