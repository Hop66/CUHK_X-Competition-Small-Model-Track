"""MotionBERT 核心模型（自包含子集）。

来源: https://github.com/Walter0807/MotionBERT (Apache-2.0)
- dstformer.py  -> lib/model/DSTformer.py + lib/model/drop.py
- action_net.py -> lib/model/model_action.py

仅保留动作识别所需的 DSTformer backbone 与 ActionNet 分类头，
避免依赖服务器 clone 完整仓库。
"""
