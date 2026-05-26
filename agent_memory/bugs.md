# 问题与风险

## 当前风险

- 最近一次 60-seed/30Hz H-FDM 训练 run `results/hfdm_training/geomapping_data1_60_h25_30hz_20260525_130644` 不应作为有效模型使用：训练停在 `epoch 1/20` 的 `3621/4169`，没有 `best.pt` 或 export。若要继续这个方向，需要重新启动训练或实现/确认可 resume 的 checkpoint 机制。
- 最新同目录 rerun `20260525_142822` 先完成 `epoch 1/20` 的 `4169/4169`，随后进程被系统 OOM-killer 以 SIGKILL 结束（`train_exit=137`），`/var/log/journal` 日志记录 `pid=345216`、`anon-rss≈11.4GB`、`Out of memory: Killed process 345216`。该事件来自系统内存耗尽，不是训练代码错误；需从内存友好配置重跑。
- Source H-FDM profile 已同步为 `nf_green_mild_noise_jerk` 控制参数并保留 60-seed `model_dir`；但 results 下原始修复 profile 仍绑定 150-seed export，后续引用时要避免把 results profile 当作 60-weight final profile。
- Source H-FDM profile 已按 no-frontend H-FDM 实际使用路径精简，但用户要求本轮不用测试；后续正式运行前仍需重新做 build/launch 或至少 `build_experiment_config` 级验证。
- `nf_green_mild_noise_jerk` 目前只 fresh 验证了 seed `424245` 修复效果，还没有完整验证 60-seed 权重 + 该 profile 的 no-frontend 多 seed 结果。
- `nf_green_tight_noise_jerk` 在 3-seed 快筛时表现很好，但补齐 5 seeds 后 seed `424245` 跑出 `40.52m` path、`89.7s` arrival，说明收紧采样噪声会引入偶发大绕行风险。
- no-frontend H-FDM 的终点捕获存在偶发失败风险：v4 的 tight sampling noise `[0.45,0.16,0.22]` 让 seed `424245` 在终点附近错过捕获后继续前冲；已验证 mild sampling noise `[0.50,0.19,0.25]` 可修复该 seed，但还未完整复验 5 seeds。
- `hfdm_no_frontend_tuned_stability.yaml` v1 虽显著降低 `wz` 换向和 jerk，但 seed `424242` 到达时间从 `50.8s` 增至 `59.6s`，不应直接作为最终调参。
- 旧 60 权重的 no-frontend H-FDM 虽 5/5 到达，但平均 arrival `55.30s`，比 150 权重 no-frontend 的 `46.52s` 慢，且 heading change/横向速度更大。

## 已知问题

- 旧 H-FDM baseline 缺少控制平滑/横向/角速度/jerk 正则和 command filter；当前 source profile 已加入这些项，但还未完成 60-seed 权重 + 精简 profile 的多 seed 复验。
- `tools/run_hfdm_model_seed_sweep.py` 复用 four-way 脚本内部 `_run_trial`，底层 per-run tag 仍包含 `hfdm150` 字样；以 sweep summary 的 `model_dir` 和 profile 为准判断实际权重。
- v4 的 `nf_green_tight_noise_jerk` seed `424245` 在 step `548` 已接近目标约 `0.65m`，但保持较高前向速度并越过目标；之后因不允许倒车、终点捕获弱、后续姿态保护未重新触发，绕到目标后方约 `6.9m` 后才回头。

## 失败尝试

- `results/hfdm_training/geomapping_data1_60_h25_30hz_20260525_130644` 训练未完成；日志没有显式 Python 异常或 OOM，最后记录为 `epoch 1/20`、`3621/4169`、loss 约 `0.14`，更像外部中断/停止而不是训练代码报错。
- 在 sandbox 内直接运行 `tools/geomapping_nav_trial.py` 会因 `~/.ros/log` 只读或 DDS socket 权限失败；实时 ROS/MuJoCo 验证需要设置 `/tmp` 日志目录并在非 sandbox 权限下运行。
- `nf_green_tight_noise_jerk_terminal` 只改 `final_controller.disable_when_local_costmap=false` 的方向已失败：seed `424245` 动态复验超时，最终距目标约 `14.80m`，不应继续推广。
- `nf_filter_soft` 在 seed `424243` 只有 wrapper reach/odom 结果，缺少 `native_run/summary.json`，离线诊断按未到达处理，不应作为候选结论依据。

## 需用户确认

- 后续是否先跑 5 seeds 还是直接扩展到更大的 final seed sweep；当前已确认 H-FDM 不需要跑 frontend 版本。

## 已关闭

- 已确认 no-frontend H-FDM 的 `planning_goals.csv` 对 seed `424242` 等成功 run 是恒定最终目标，不是隐藏中间路径约束导致的绕行。
- 已确认旧 60 权重可以在 CUDA runtime 连续推理，不需要重新导出或重训即可跑导航。
