我把仓库当前 `main` 分支、`doc/`、关键训练/验证脚本，以及官方 CUHK-X 基线和一些同类 HAR / 轻量视频识别 / Domain Generalization 项目交叉检查了一遍。结论比较明确：

**现在项目的“停滞”很大一部分不是因为已经把模型能力榨干，而是因为实验体系已经被污染：不同协议混比、验证集反复调参、实现 bug、full 模型被误认为 fold 模型，以及 leaderboard 反馈参与方法选择。**

所以我不建议继续按照 `doc/METHODOLOGY_SUMMARY.md` 里“哪些方法已被证伪/哪个方法突破”的结论往下走。应该先把实验体系重置。

## 1. 现有成绩应该怎么理解

文档目前把 `main_s42 + th_nf32full + prob_avg + flip` 的 **LB 0.75124** 当锚，并把骨架 hard-pair gate 的 **0.75621** 当当前最佳。

这两个 leaderboard 数字如果提交记录准确，本身可以视为“真实发生过的提交分数”。

但不能进一步推导：

> “骨架 gate 稳定提升了 0.5pt”  
> “某方法已经被永久证伪”  
> “训练已经达到瓶颈”  
> “双塔难对训练提升 5.18pt”

因为这些**方法效果的归因**存在严重实验问题。

而且原始 95KB 实验日志已经误覆盖，现在的 `RUNLOG_remote.md` 明确说明是从记忆和片段重建的，不是完整实验历史。

因此我会把目前证据分成两档：

|内容|当前可信程度|
|---|---|
|某 CSV 在 Kaggle 上实际拿到 0.75124 / 0.75621|较高|
|“为什么拿到这个分数”|较低|
|fold 上的 `+0.x/+x.x pt`|很多需要重跑|
|“永久关闭某方向”|基本不应继续采信|
|最新 `+5.18pt` 双塔 breakthrough|**无效实验，必须重算**|

---

# 2. 我找到的几个关键实验问题

### A. `split_by_subject()` 本身没犯最基础的泄漏错误

这个要先说明清楚。`src/split.py` 确实按照 subject 分组，并显式 assert train/val subject 不相交，因此不是“随机 clip 切分导致同一人同时出现在 train/val”的低级泄漏。

真正的问题发生在**split 之后的实验选择过程**。

---

### B. 一个 FULL 模型被当成了“fold0 OOF 模型”

这是目前最严重的问题之一。

`check_th_main_inprotocol.py` 写着：

> `main_full/r2plus1d34_depthir_full_seed42.pth` 是 fold0 模型，拿 fold0 validation subjects 去评估属于“干净 OOF”。

但训练脚本实际上明确：

> `FULL 全量训练（不分折）`  
> `if args.full: tr_clips = clips`。

也就是说这个 checkpoint **已经训练过所谓 fold0 validation subjects**。

为什么会产生误会？

因为：

```text
outputs/main_full/r2plus1d34_depthir_full_seed42.pth
```

经过 `quantize_pack.py` 后，即使只有一个 checkpoint，它也按列表下标命名为：

```text
main_s42_fold0_int5.pth
```

这里的 `fold0` 只是“第 0 个输入 checkpoint”的编号，并不表示模型是 fold0 训练出来的。相关打包命令明确指向 `main_full/...full_seed42.pth`。

因此：

**凡是拿 `main_full` / `main_s42_fold0_int5` 回头在训练集 fold0 上做“OOF 验证”的实验，一律作废。**

---

### C. 骨架 gate 的 fold 验证和最终 test gate 根本不是同一个算法

`gate_fold_check_skel_only.py` 看起来是在验证：

> 对某个 hard pair `(a,b)` 训练二分类 GBDT。

但实际上训练数据 `mti` 没过滤 `(a,b)`：

```python
for a, b in HARD:
    for c in tr_c:
        mti.append(...)
    clf.fit(...,
            [1 if action == a else 0 ...])
```

而 `tr_c` 包含**所有 hard-pair 类**。

所以 `(13,12)` 的所谓二分类器实际上学的是：

> `class 13 vs 所有其它 hard classes`

而不是：

> `13 vs 12`

但最终生成 test CSV 的 `make_test_gate_csv.py` 却正确过滤了：

```python
if action in (a,b)
```

也就是 test 用的是真正 `a vs b` 分类器。

因此文档宣称的：

> “fold +0.94pt，机制与 test 完全一致”

实际上不成立。

更进一步，`tau`、`conf_lo/conf_hi`、甚至 `pairs_keep` 都是在相同 OOF folds 上看结果之后挑的，再拿这些 folds 报效果，是典型的**model-selection bias**。

所以 0.75621 这个提交值得保留，但：

**不能据此说 skeleton gate 已被严格证明稳定 +0.5pt。**

它完全可能包含 public-LB 运气成分。

---

### D. 最新“+5.18pt 双塔大突破”存在 double-softmax bug

这是刚刚加入最新 commit 的实验。

文档写：

> baseline dual 0.6355 → hard-pair dual 0.6872，提升 +5.18pt。

但 `hardpair_dual_fold.py` 做了：

```python
bm = softmax_logits(...)
bt = softmax_logits(...)
```

已经 softmax 一次。

进入：

```python
eval_fusion(pm, pt):
    fu = sf(pm[k]) + sf(pt[k])
```

又 softmax 一次。

也就是实际比较的是：

**softmax(softmax(logits))**

而提交使用的是正常的：

**softmax(logits)**

这会严重压平概率分布、改变两个模型的相对校准和融合决策。

所以当前写进文档的：

> `0.6355 → 0.6872 (+5.18pt)`

**不能作为有效实验。**

而且这个 fold 实验的 thermal 是 **16 frames**，目前真正的 anchor thermal 是 **32 frames**。

即使没有 double-softmax，它也不是当前 0.75124 锚链的严格 AB。

---

### E. 即便按文档自己的数字，“+5.18pt”也选错了 baseline

同一个表中：

- baseline main = **0.6674**
    
- baseline main+thermal = **0.6355**
    
- hard-pair main+thermal = **0.6872**
    

一个融合方案比单 main 还低 3.2pt，本身已经说明 fusion baseline 有问题。

所以即便算法实现完全正确，真正有意义的增益也最多应该先看：

`0.6872 - 0.6674 ≈ +1.98pt`

而不是 +5.18pt。

---

### F. “512d feature 不如 40d logits”的决定性结论也不成立

`extract_oof_logits.py` 启用 flip 时：

```python
o = model(x)
o = o + model(flip(x))
```

所以 logits 是两个 view 的结果。

但 encoder hook 在第二次前向后被覆盖，最后保存的 512d feature 只是**flip view 的 feature**。

于是目前文档比较的是：

- 40d：original + flip
    
- 512d：flip only
    

这不是同协议。

因此文档里：

> “512d 跨折 0.473 < softmax 0.717，所以 feature 已无可利用信息”

这种结论应该撤回。

---

### G. GRL subject-invariant 代码目前也有实现问题

`train_grl.py` 试图这样修改 dataset：

```python
train_ds.__getitem__ = new_getitem
```

但是 Python 的 `obj[idx]` 对 `__getitem__` 这种 special method 是从**类**上解析，不是正常从 instance attribute 查找。所以这种 monkey patch 并不能可靠改变 DataLoader 的行为。

而原 dataset 第三个返回值实际上是 `clip.subject` 字符串。

另外 GRL 内部已经乘了一次 λ，外部 subject loss 又乘 λ，相当于 backbone 的 adversarial 梯度强度是近似 **λ²**，也不是标准 DANN 实现。

所以 GRL 这个方向目前并没有得到一次可信的验证。

---

# 3. 为什么现在尤其不能继续追 public LB

官方比赛规则对这个项目非常关键。

Small Model Track 要求：

- ≤100MB
    
- conventional CNN/RNN/Transformer
    
- **no large pretrained backbones**
    
- Top 15 还要做代码复现与现场 inference
    
- 最终 Top 6 会在 **brand-new private dataset** 上现场推理。([UbiComp/ISWC 2026](https://www.ubicomp.org/ubicomp-iswc-2026/cuhk-x-competition/?utm_source=chatgpt.com "CUHK-X Competition - UbiComp/ISWC 2026"))
    

所以“针对现在 405 个 test clips 调 gate / pair / threshold，然后看 LB”战略价值其实很低。

真正应该最大化的是：

> **unseen subject + unseen acquisition/domain 的泛化。**

此外你们大量实验明确使用：

```text
ig65m_r2plus1d34.pth
```

例如目前 main/thermal 训练脚本都是这样初始化的。

我不能仅凭规则文字断言 R(2+1)D-34 + IG65M 一定违规，但官方明确写着 **no large pretrained backbones**，而 IG65M 大规模视频预训练显然至少存在合规风险。([UbiComp/ISWC 2026](https://www.ubicomp.org/ubicomp-iswc-2026/cuhk-x-competition/?utm_source=chatgpt.com "CUHK-X Competition - UbiComp/ISWC 2026"))

**这个问题建议尽快向主办方书面确认。**

同时应该建立一条不依赖 IG65M 的 compliance-safe 方案。

---

# 4. 我建议重新建立这样的公平实验协议

CUHK-X 官方 baseline 特别指出数据其实来自两个环境：

- Environment A：user 1–15
    
- Environment B：user 16–30。
    

而比赛恰好：

- train：1–9 / 16–24
    
- hidden：10–11 / 25–26
    

也就是 hidden test 是 **两个环境各两个新 subject**。

目前随机把 18 个 subjects 切成 3×6，并不能很好模拟真实测试。

我建议新协议是：

|层级|新协议|
|---|---|
|Locked audit set|永久留 4 subjects：Env-A 2 人 + Env-B 2 人；在模型家族锁定前绝不查看|
|Development|其余 14 subject 做 environment-balanced Group CV|
|超参|hard-pair、threshold、fusion α、epoch 全部只能在 inner CV 选|
|Outer result|outer subject 从未参加 pair/threshold/epoch 选择|
|Seed|至少 42 / 2024 / 777|
|inference|crop / frames / flip / quantization 与最终 submission 完全一致|
|report|overall Acc + per-subject Acc + macro-F1 + worst-subject + mean/std|
|LB|只做最终 sanity check，不参与方法搜索|

最重要的是：

**一个实验只能改变一个变量。**

例如：

`main16 + thermal32`

和：

`main16-hardpair + thermal32-hardpair`

必须满足 seed、epoch、sampler、pretrain、crop、TTA、quantization、fusion 完全相同。

---

# 5. 网上同类型项目里，我认为最值得移植的方向

## 第一优先级：换一个真正轻量的 temporal backbone，换取 ensemble diversity

现在两个 R2+1D 模型几乎吃掉整个 100MB 预算。

相比之下：

- **TSM-MobileNetV2** 只有约 **2.736M parameters**，Temporal Shift 本身零新增参数、零新增 FLOPs。([GitHub](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/README.md?utm_source=chatgpt.com "mmaction2/configs/recognition/tsm/README.md at main · open-mmlab/mmaction2 · GitHub"))
    
- **X3D-XS/S/M** 约 **3.8M parameters**，X3D-M 仍只有 3.8M。([GitHub](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md?utm_source=chatgpt.com "SlowFast/MODEL_ZOO.md at main · facebookresearch/SlowFast · GitHub"))
    

这对这个比赛非常有价值。

当前策略是：

> 两个非常相似的巨大 R2+1D → 77MB

新的策略可以变成：

> main TSM/X3D
> 
> - thermal TSM/X3D
>     
> - skeleton tiny model
>     
> - 不同 temporal scale / seed ensemble
>     

仍可能远低于 100MB。

**我最优先会跑 TSM-MobileNetV2。**

原因不是它单模型一定超过当前 R2+1D，而是 ensemble 真正需要的是**误差互补**，不是两个巨大且高度相关的模型。

---

## 第二优先级：真正针对 subject/environment shift，而不是继续针对 hard pairs

官方 CUHK baseline 本身已经包含 cross-subject、contrastive learning 和 environment-aware 实验。

这一点与你们现在的重点区别很大。

比较值得尝试的是：

**MixStyle**：在中间 feature statistics 间做 cross-domain mixing，几乎没有推理成本，专门用于 domain generalization。([GitHub](https://github.com/KaiyangZhou/mixstyle-release?utm_source=chatgpt.com "GitHub - KaiyangZhou/mixstyle-release: Domain Generalization with MixStyle (ICLR'21) · GitHub"))

再配合：

**GroupDRO**：把 subject 或 environment 当 group，直接优化 worst-group performance。DomainBed 已提供成熟实现，并特别区分正常 leave-one-domain model selection 和 oracle selection——这正好对应你们目前 validation-selection 混淆的问题。([GitHub](https://github.com/facebookresearch/DomainBed?utm_source=chatgpt.com "GitHub - facebookresearch/DomainBed: DomainBed is a suite to test domain generalization algorithms · GitHub"))

以及：

**SWAD**：通过寻找 flatter minima 改善 domain generalization，而且官方实现就是 leave-one-domain-out evaluation。([GitHub](https://github.com/khanrc/swad?utm_source=chatgpt.com "GitHub - khanrc/swad: Official Implementation of SWAD (NeurIPS 2021) · GitHub"))

对这个比赛而言，我认为：

**MixStyle > GroupDRO > SWAD**

值得依次测试。

---

## 第三优先级：把 GRL 修好，而不是放弃 subject-invariant learning

跨人 HAR 本身就有专门工作。

例如 GILE 的目标就是：

> 不访问 target-person 数据，学习 domain-agnostic / person-invariant representation。

并有公开实现。([GitHub](https://github.com/Hangwei12358/cross-person-HAR?utm_source=chatgpt.com "GitHub - Hangwei12358/cross-person-HAR: Code for our AAAI-2021 paper \"Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition\". · GitHub"))

2026 年新的 HAROOD benchmark 更进一步，专门标准化了 cross-person、cross-position、cross-time、cross-device 等 OOD HAR 场景，并实现 16 种 generalization 算法。([GitHub](https://github.com/AIFrontierLab/HAROOD?utm_source=chatgpt.com "GitHub - AIFrontierLab/HAROOD: [KDD'26] A modular and reproducible benchmark framework for studying generalization in sensor-based human activity recognition. · GitHub"))

所以 subject-invariance 本身并不是错误方向。

现在的问题是**你们 GRL 代码没得到一次干净实验**。

把 Dataset wrapper 和 λ 修好以后，我会优先测试：

`ERM baseline → GRL/DANN → MixStyle → MixStyle+GRL`

而不是继续增加 hard-pair 规则。

---

# 6. 长尾问题也还没有被真正解决

目前文档把一次 CB Loss 失败之后，很大程度上把 reweighting 方向降级了。

但现代 long-tail 方法的核心经验恰好是：

> **不要在 feature learning 阶段猛烈重权。**

cRT/LWS 的经典结果就是：先正常学习 representation，然后冻结 backbone，单独重训 classifier。([GitHub](https://github.com/facebookresearch/classifier-balancing?utm_source=chatgpt.com "GitHub - facebookresearch/classifier-balancing: This repository contains code for the paper \"Decoupling Representation and Classifier for Long-Tailed Recognition\", published at ICLR 2020 · GitHub"))

这非常适合你们，因为：

- 不破坏 IG65M / 自监督得到的 representation；
    
- 几乎不增加模型大小；
    
- 和现在“训练一加东西就掉 2–7pt”的现象高度吻合。
    

我会测试：

**cRT → LWS → Balanced Softmax → LDAM-DRW**

Balanced Softmax 是**训练 loss**，不是你们已经失败的 post-hoc prior/logit adjustment；官方实现很简单。([GitHub](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification?utm_source=chatgpt.com "GitHub - jiawei-ren/BalancedMetaSoftmax-Classification: [NeurIPS 2020] Balanced Meta-Softmax for Long-Tailed Visual Recognition · GitHub"))

LDAM-DRW 也有成熟公开实现。([GitHub](https://github.com/kaidic/LDAM-DRW?utm_source=chatgpt.com "GitHub - kaidic/LDAM-DRW: [NeurIPS 2019] Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss · GitHub"))

---

# 7. 一个被低估的方向：在 CUHK-X 自己的数据上做 SSL

官方 CUHK-X baseline 已经提供 SimCLR cross-subject 训练代码。

这对 Small Model Track 特别重要，因为可以：

**不依赖“大规模外部 pretrained backbone”，而用比赛 training data 自己预训练。**

例如：

```text
Depth+IR view A
Thermal view B
同一 clip = positive
不同 clip = negative
```

先做 cross-modal / temporal contrastive pretraining，再 supervised fine-tune。

这样既可能提高 unseen-subject representation，也能降低 IG65M 合规风险。

这比继续用 skeleton 手工 feature 去改 3 个 test predictions 更符合最终现场 private test 的目标。

---

# 8. 我建议现在停止的方向

暂时不要继续做新的：

- test-specific hard-pair gate；
    
- 手调 pair whitelist；
    
- conf/margin threshold sweep；
    
- public LB 驱动的 post-hoc override；
    
- 新的 per-class correction；
    
- 基于 405 test prediction 分布反推 prior；
    
- “fold 正一点就 full train + LB”的流程。
    

它们不是完全没有价值，而是**当前实验基础不允许判断它们到底有没有价值**。

---

# 9. 我会按这个顺序重新启动项目

|优先级|实验|原因|
|---|---|---|
|**P0**|修 double-softmax / gate pair classifier / full-vs-fold 命名 / feature TTA / GRL|不修这些，后面数字没有意义|
|**P0**|建 locked env-balanced subject audit split|阻止继续 validation overfit|
|**P1**|重建 main16 + thermal32 公平 baseline|得到新的可信锚|
|**P1**|TSM-MobileNetV2 main / thermal|极低参数，ensemble 潜力最高|
|**P1**|X3D-XS/S main / thermal|第二条轻量视频骨干|
|**P1**|MixStyle|几乎零 inference cost，直接解决 domain shift|
|**P2**|修正版 GRL / subject adversarial|直接针对 unseen subjects|
|**P2**|cRT / LWS|长尾且不会破坏 backbone|
|**P2**|Balanced Softmax / LDAM-DRW|比现有 naive CB loss 更合理|
|**P2**|CUHK-X-only SimCLR / cross-modal SSL|泛化 + 合规安全|
|**P3**|calibrated main/thermal fusion|inner-CV 学 temperature + 单个 α|
|**P3**|tiny skeleton member|只有在独立 outer split 显示互补后才加入|

如果必须从中只选**三个最值得立刻投入 GPU 的方向**，我会选：

**TSM-MobileNetV2 → MixStyle → 修正版 GRL。**

而在它们之前，先把 P0 的评估 bug 修掉。

---

还有一个很重要的判断：**当前仓库并不是“0.75 已经做到极限”**。现阶段最多能说“当前 R2+1D + 当前不稳定实验体系停在 0.75 左右”。官方比赛最终要跑全新 private data，当前正好应该从“调 test predictions”切回“跨人、跨环境、轻量多样模型”的路线。([UbiComp/ISWC 2026](https://www.ubicomp.org/ubicomp-iswc-2026/cuhk-x-competition/?utm_source=chatgpt.com "CUHK-X Competition - UbiComp/ISWC 2026"))

如果你愿意，我下一步可以直接基于这个 GitHub 仓库开始做第一轮代码整改：**先修 `hardpair_dual_fold.py`、gate 验证、`check_th_main_inprotocol.py`、GRL 和统一的 fair-eval 脚本，并把修改整理成一个独立 branch/PR**。这样后续所有新实验就可以建立在同一个可信协议上。