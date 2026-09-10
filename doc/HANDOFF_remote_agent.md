# 任务交接 — CUHK-X 小模型赛道（给远程窗口的 Copilot Agent）

> 从本地会话交接。请先完整读本文件，再动任何脚本/任务。

## 一句话目标
把测试集提交（405 clip → CSV `path,prediction`）从当前采信的 **LB 0.73** 往上冲，至少保住 0.73。
当前有一次**进行中的验证**：骨架「真·重采样」是否能让被判负的 SM-dual 翻盘（见 A 待办）。

## 项目背景（3 点）
- 竞赛：CUHK-X Multimodal HAR / UbiComp26 小模型赛道；40 分类、跨被试、模型打包后 ≤100MB。
- 现有基线链：`main_s42(int5) + thermal_s42(int5)`，`prob_avg + flip TTA` → **LB 0.73**；`main_s42` 单模态 0.70646。
- 其余模态/dual 全部实测判负（见教训 3），**提交的核心组合仍是 main+thermal**。

## 环境速查
- 服务器（本窗口即远程）：所有代码在 `~/Multimodal`；conda env `cuhk_x`：
  `source $(conda info --base)/etc/profile.d/conda.sh && conda activate cuhk_x`
- Slurm：sbatch 头统一 `-p Students --qos=qos_stu_long --gres=gpu:A100:1`；**sbatch 文件必须 LF**（无 CR）。
- 数据：训练 `~/Multimodal/data/Training/HAR`（+ `data/Skeleton`）；测试 `~/Multimodal/data/Testing/data/small_model_track_test`（405 clip，各含 Depth_Color/IR/Thermal/Skeleton/predictions）。
- 关键 pack/ckpt（`~/Multimodal/outputs/`）：
  - `pack/main_s42_fold0_int5.pth`（main 全量 seed42 int5）→ 0.706 单 / 0.73 融合
  - `pack/thermal_s42_fold0_int5.pth`（0.73 链 thermal）
  - `pack/main_dual_s42_fold0_int5.pth`（SM-full dual int5，LB≈0.637）
  - `main_dual/main_S{,_M}_fold{0,1,2}.pth`（fold 划分 dual，LB≈0.622）
  - `motion_cache.pkl`（训练骨架运动特征缓存，29 维）

## 已确诊教训（勿再踩）
1. **推理 crop 必须用测试集 bbox**：main→`bbox_test.json`、thermal→`bbox_thermal_test.json`；训练段才用 `bbox_train.json`/`bbox_thermal_train.json`。用错=测试侧假崩（曾致 0.55/0.59 假数据）。
2. `quantize_pack` 对 `numel<4096` 张量（含学习 α）不量化、保持 fp32。
3. **骨架 dual（MotionNet+α 并入 main/thermal）在 LB 判负**：fold val 三折全正（+0.6/2.3/2.8），但跨采集管线 LB：SM-full+thermal=0.637、SM fold 三折+thermal=0.62189（含「速度项×0.5 cadence 校准」后仍一字不变）→ motion 跨域是加噪；**「val≈LB」只在同管线成立，跨域不能外推**。
4. TTA4/prior 只在本地验证过（+1.29/+0.43），LB 未单独 AB；flip TTA 属 0.73 链组成部分。

## 当前待办（优先级顺序）
- **A【进行中 · 用户坚持】真·重采样验证**：目前只做过「速度项 ×0.5」标量校准（域对齐 2.09→1.04 成功，但 LB 不变；原因疑为 MotionNet 首层 BN 对输入 scale 鲁棒）。待验证路线：把**测试骨架序列重采样到训练 cadence/总时长**后再提取特征（改 `src/skeleton_motion.py::extract_motion_features` 支持 `resample` 参数，传入测试序列预升/降采样），或训练与测试统一按物理时间（真实 dt）算速度。跑完后对比 LB：若仍不变 → dual 线彻底关闭；若回升 → dual 复活，值得继续（fold 0.6652 泛化是真的）。
- **B【升级候选】main 多 seed / SWA**：`ls outputs/main_full/` 看 seed2024/777 是否已训；有 → `quantize_pack --name main_avg --average --checkpoints outputs/main_full/r2plus1d34_depthir_full_seed{42,...}.pth --bits 5`，再与 thermal 融合出 CSV。
- **C【前置】服务器代码同步**：跑之前先确认服务器上 `scripts/ensemble_inference.py`、`src/skeleton_motion.py`、`scripts/selftest_*`、`scripts/submit_fold_dual.sbatch`、`scripts/main_dual_full.sbatch` 是本地最新版（Local 会话刚更新过），否则先 `scp`/`git` 同步。
- **D** thermal 单独增强（融合里的隐形主力）。

## 判读/验真基准
- 最终提交必须 ≥ 0.73（地板）。新组合先自验再上 LB。
- 推理链路金标准：`python scripts/selftest_main_dual_infer.py --ckpt outputs/main_dual/main_SM_fold0.pth`
  → 应复现 learned=0.6652 / static=0.6588（覆值和链路正确性）。
- 域对比诊断：`python scripts/diag_motion_domain.py --limit 200 [--speed_scale 0.5]`（速度幅值比≈1 表示域对齐）。

## 命令速查（出 CSV）
- 单模态：`python scripts/ensemble_inference.py --main outputs/pack/main_s42_fold0_int5.pth --quantize --main_crop bbox_test.json --flip_tta --prob_avg --output X.csv`
- 融合：上面再加 `--thermal outputs/pack/thermal_s42_fold0_int5.pth --thermal_crop bbox_thermal_test.json`
- dual：`--main_dual ... --test_speed_scale 0.5`

## 参考
- `doc/SSH_usage.md`（SSH/Remote-SSH 说明）；`doc/CUHK-X_*md`（方案/实现/优化）；仓库记忆 `/memories/repo/cuhkx-2026.md`（正在被维护，可读不可乱删）。

---

# 远程会话交接补记（2026-09-03 · 待办 A/B 已执行）

## A【已裁决 · dual 线关闭】真·重采样验证
- **实现**：`src/skeleton_motion.py::extract_motion_features(kp, resample=M)` 新增 kp 级真·重采样；
  `scripts/ensemble_inference.py` 新增 `--test_resample f`（M=round(N·f)）接入 dual 路径。
- **域诊断新发现**（`diag_motion_domain.py --resample_scan`）：训练运动缓存 N帧 mean=29.0 vs 测试 mean=26.3（**帧数相近**），
  但速度 dims 测试/训练幅值比=2.086 → **kp 升采样×2.0 → 比 1.02 ✓**。物理解读：测试每帧跨越真实时间≈训练 2 倍
  （测试实际更「稀」），对齐要**升采样**而非旧推断的降采样×0.5（旧「测试≈训练×2 fps」方向疑似反了）。
- **AB 裁决**（job 53307，CSV md5 逐字节比对）：
  - full-SM：`base(ss0.5) == rs20(升采样×2) == rs20+ss0.479(双对齐)`（同一 md5）→ **405 clip argmax 全不变**
  - foldavg：`base == rs20`（同一 md5）→ **全不变**
  - 自验基准复现：learned=0.6674 / static=0.6663 / motion-only=0.2853 / α(fold0)=0.548（金标准≈0.6652，OK）
  - 结论 **dual 线关闭**：MotionNet 对 motion 输入域/scale 鲁棒，无论速度项标量缩放还是 kp 级重采样，
    LB 一字不变（0.637 / 0.62189 不可能翻盘）。**别再投入改骨架运动编码**；后续 dual 若想救只能从
    static 侧 / 融合权重入手，且要先有 ≥0.73 的证据链否则免谈。

## B【候选已就绪 · 待 LB】main/thermal SWA（多 seed 平均）
- 登录节点 CPU 直接打包（quantize_pack 不占 GPU）：
  - `outputs/pack/main_avg_int5.pth`（seed42+2024+777 SWA，40.2MB）
  - `outputs/pack/thermal_avg_int5.pth`（seed42+777 SWA，40.2MB）
- 同 job 4 单元融合（job 53308，prob_avg+flip，全部 int5）：
  - `sub_B_mains42_ths42.csv` 参照（≈0.73 链复现）
  - `sub_B_mainavg_ths42.csv`（main SWA，vs 参照翻转 88/405）
  - `sub_B_mains42_thavg.csv`（thermal SWA，翻转 66/405）
  - `sub_B_mainavg_thavg.csv`（双 SWA，翻转 142/405，收敛 33 类）
- **待 LB**：建议按 参照→单 SWA→双 SWA 顺序测；任一 >0.73 即替换 0.73 链。

## 记录/产物清单
- 新脚本：`scripts/ab_dual_resample.sbatch`、`scripts/ab_main_swa_fusion.sbatch`（均可重放）
- 诊断：`scripts/diag_motion_domain.py --resample_scan`（kp 级重采样因子扫描）
- 记忆：`/memories/repo/cuhkx-2026.md`、`/memories/session/cuhkx-remote-plan.md`

