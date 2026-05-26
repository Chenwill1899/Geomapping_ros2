# 当前任务进度

## 当前任务

- 任务：记录最终 H-FDM no-frontend 实验和科研图表计划，等待后续运行复验。
- 成功标准：后续用 60-seed H-FDM 权重 + `nf_green_mild_noise_jerk` profile 完成 no-frontend final sweep；主表能报告成功率、到达时间、路径长度、终点距离、控制平滑性和运行时；关键图能解释 seed `424245` 终点捕获修复。
- 范围边界：H-FDM 不跑 frontend 版本；不重训、不改 MPPI 核心逻辑；nominal frontend 只可作为传统基线，不作为 H-FDM 条件。
- 停止条件：最终组合在代表 seeds 上无法保持到达，或 profile 绑定/运行环境问题导致无法获得 `native_run/summary.json`。

## 已完成

- 已检查最近一次训练目录 `results/hfdm_training/geomapping_data1_60_h25_30hz_20260525_130644`：转换和 dataset build 已完成，`dataset_info.json` 记录 `480` train / `120` val episodes、`1067079` train sequences、`254983` val sequences。
- 已确认该训练未完成：`logs/03_train.log` 停在 `epoch 1/20` 的 `3621/4169`，没有 `run/checkpoints/best.pt`、没有 `logs/04_export.log`、没有 `export/fdm_ts.pt` 或 CUDA trace export。
- 已新增 `docs/mppi_fdm_constraints.html`，用 MathJax HTML 写出无 FDM nominal MPPI 与 H-FDM MPPI 的采样、控制边界、rollout、目标/进度/heading/control 正则、显式 obstacle/local-costmap/terrain 约束和 H-FDM learned risk 公式。
- 已将 `docs/mppi_fdm_constraints.html` 改成更清晰的对比结构：页面开头直接并排展示 \(S_i^{nom}\) 与 \(S_i^{fdm}\)，并用逐项表说明 FDM 如何把手写 rollout + \(J_{obs}/J_{costmap}/J_{terrain}\) 替换成 high_level_fdm 预测 + \(J_{risk}^{fdm}\)，同时保留硬控制边界和控制正则。
- 已补齐并读取 `agent_memory/` 模板文件。
- 已确认 20260525_142822 重跑进入 `epoch 1/20`，完成 `4169/4169` 后被系统 OOM-kill（`train_exit=137`），`run/checkpoints` 仍未产出，属于外部资源不足导致中断。
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
- 已记录最终实验计划：用户选择旧 60-seed/600-episode H25 export 作为最终权重依据，选择 `results/mppi_tuning/20260524_hfdm_no_frontend_seed424245_fix/profiles/nf_green_mild_noise_jerk.yaml` 作为最终 MPPI 控制 profile；H-FDM 最终不跑 frontend 版本。
- 已明确论文图表计划：主结果表、60 vs 150 权重选择表、no-frontend profile 消融表、轨迹 overlay、seed424245 distance-to-goal 终点捕获图、控制平滑性图、运行时开销表。
- 已新增 adaptor launch 入口 `src/ausim_geomapping_adapter/launch/ausim_scout_mppi_hfdm_no_frontend.launch.py`，当前引用 source H-FDM profile `mujoco_rviz_goal_hfdm_h25.yaml`，并设置 `mppi_controller=learned_hfdm_h25`、`use_frontend=false`。
- 已将 source H-FDM profile `src/mppi_controller/configs/mujoco_rviz_goal_hfdm_h25.yaml` 同步为 `nf_green_mild_noise_jerk` 控制参数，同时保留 60-seed H-FDM 权重 `model_dir=/home/mexxiie/prj/high_level_fdm/runs/geomapping_data1_h25/export`。
- 已将 `src/ausim_geomapping_adapter/launch/ausim_scout_mppi_topdown_heightmap.launch.py` 对齐为与 `ausim_scout_mppi_hfdm_no_frontend.launch.py` 相同的 H-FDM no-frontend 启动参数，仅额外设置 `height_source=mujoco_topdown`。
- 已按 H-FDM/no-frontend 实际代码路径精简 `src/mppi_controller/configs/mujoco_rviz_goal_hfdm_h25.yaml`：删除 external/global/local/final 传统前端约束子参数、显式动态障碍约束块、reactive avoidance、local-costmap nominal cost 参数、terrain/obstacle cost 和重复 learned-risk 字段；保留 learned rollout、learned risk、goal/progress/heading cost、控制平滑正则、command filter、rotate-then-translate 和 local costmap 模型输入。

## 进行中

- 无需继续重跑该会话；当前 rerun 已结束（SIGKILL/oom），等待下一次按新参数重启。

## 待处理

- 只跑 no-frontend H-FDM final sweep；不跑 H-FDM frontend。
- final sweep 后用 `tools/analyze_hfdm_sweep.py` 生成 diagnostics，并据此整理科研图表和表格。
- 论文主表建议比较 `frontend nominal MPPI`、`no-frontend nominal MPPI`、`no-frontend H-FDM final`；其中 `frontend nominal MPPI` 只是传统前端基线。

## 验证

- 已验证训练已重新开始：`python3 -m hfdm.cli.train` 进程存在，`logs/03_train_rerun_20260525_142822.log` 已进入 `epoch 1/20`，约 `93/4169` step；GPU 占用约 `474MiB/4096MiB`、utilization 约 `19%`。
- 已验证：当前没有 H-FDM/训练相关 Python 进程；最近训练日志未包含 `Traceback`、`RuntimeError`、`Killed`、OOM 等显式错误文本。现有证据支持“训练被外部中断/停止，未产出有效 checkpoint”，不是模型 loss 数值失败或导出阶段失败。
- 已验证：`resume_pipeline_20260525_142822.log` 记录 `train_exit=137`；`journalctl -k` 对应内核日志显示 `task=pt_main_thread,pid=345216` 在 `2026-05-25T14:40:13+08:00` 被 `Out of memory: Killed process`，且 `OOM-killer` 释放该进程，确认进程是内存耗尽杀死而非代码异常。
- 本轮公式 HTML 未跑测试；已按 `mppi_omni_torch.py`、`mppi_omni_high_level_fdm_torch.py`、`high_level_fdm_runtime.py` 和当前 H-FDM profile 做源码级核对。H-FDM 文档中特别标明预测状态使用 model `applied_twist`，但 accel/jerk/lateral/yaw-rate 正则仍来自 inherited response rollout 的 `real_controls`。
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
- 已验证新增/更新后的 H-FDM source config 与 launch：`PYTHONPATH=src/mppi_controller python3 -m pytest -q src/mppi_controller/test/test_cuda_mppi_defaults.py` 通过 `8 passed`；`python3 -m py_compile src/ausim_geomapping_adapter/launch/ausim_scout_mppi_hfdm_no_frontend.launch.py` 通过；`colcon build --packages-select mppi_controller ausim_geomapping_adapter --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release` 通过；设置 `ROS_LOG_DIR=/tmp/ros_logs` 后 `ros2 launch ausim_geomapping_adapter ausim_scout_mppi_hfdm_no_frontend.launch.py --show-args` 通过；直接检查 include 参数为 `mppi_profile=<mppi_controller/configs/mujoco_rviz_goal_hfdm_h25.yaml>`、`mppi_controller=learned_hfdm_h25`、`use_frontend=false`。
- 本轮 topdown launch 对齐后，用户要求不用测试；此前仅完成 `py_compile` 与 `git diff --check`，未继续跑 ROS launch/构建。`PYTHONPATH=src/ausim_geomapping_adapter python3 -m pytest -q src/ausim_geomapping_adapter/test/test_pipeline_topdown.py` 因当前 Python 路径下 `launch.conditions` 导入失败而停止，未作为功能结论。
- 本轮 H-FDM profile 精简按用户要求未运行测试、构建或 ROS 启动验证；只基于代码引用和 diff 审核判断字段有效性。

## 下一步

- 下一步直接用当前 source H-FDM profile 执行 no-frontend H-FDM final sweep；完成后更新主结果表、权重选择表、profile 消融表、轨迹图、终点捕获图、控制平滑性图和运行时表。
