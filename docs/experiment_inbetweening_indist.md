# 实验方案：分布内动作序列的关键帧 In-betweening

## 目标

给定 GT 关键帧切分，且目标动作序列已转移到训练动作序列分布内，生成关键帧对齐、光滑自然的动作序列

---

## 数据集

从时序动作生成模型的训练数据集随机选择 10 条动作序列，人工标注关键帧序号，标注的关键帧应该满足：
- 关键帧间是简单路径（近似直线）
- 关键帧间姿态差异小
- 选择一段关节变动大的时序增加关键帧密度

---

## 评估指标

### 符号定义

- $T$：序列总帧数
- $J$：关节总数
- $\mathbf{x}_{t,j}^{gt},\; \mathbf{x}_{t,j}^{pred} \in \mathbb{R}^3$：第 $t$ 帧第 $j$ 个关节的 GT / 预测三维位置
- $\Delta t$：帧间时间间隔（$\Delta t = 1 / \text{fps}$）
- $\mathcal{K} = \{t_1, t_2, \ldots, t_M\}$：关键帧的帧索引集合，$M = |\mathcal{K}|$ 为关键帧数量

### 1. 关键帧匹配程度

**关键帧整体偏移量（Keyframe MPJPE, KF-MPJPE）**：在关键帧处所有关节的平均位置误差

$$\text{KF-MPJPE} = \frac{1}{M} \sum_{t \in \mathcal{K}} \frac{1}{J} \sum_{j=1}^{J} \left\lVert \mathbf{x}_{t,j}^{gt} - \mathbf{x}_{t,j}^{pred} \right\rVert_2$$

### 2. 平滑度指标

**Jitter**：关节轨迹的三阶有限差分（jerk）的平均幅度

$$\text{Jitter} = \frac{1}{(T-3) \cdot J} \sum_{t=1}^{T-3} \sum_{j=1}^{J} \frac{\left\lVert \mathbf{x}_{t+3,j} - 3\,\mathbf{x}_{t+2,j} + 3\,\mathbf{x}_{t+1,j} - \mathbf{x}_{t,j} \right\rVert_2}{\Delta t^3}$$

---

## 思路

要完成目标，主要要素有：
1. 模型本身合成动作的质量要高
2. 模型支持关键帧约束，且约束效果好
3. 模型在约束效果好的前提下，还能维持高质量

---

## 可用模型

| 角色 | 模型 | 训练时关键帧条件 | 仓库 |
|------|------|----------------|------|
| 里程碑模型 | **MDM**（ICLR 2023） | 否 | [GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model) |
| 动作质量 SOTA | **MoMask**（CVPR 2024） | 否（masked modeling 训练时 mask pattern 覆盖稀疏散布） | [EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes) |
| 关键帧可控 SOTA | **CondMDI**（SIGGRAPH 2024） | 是（imputation ckpt） | [setarehc/diffusion-motion-inbetweening](https://github.com/setarehc/diffusion-motion-inbetweening) |

### MDM（ICLR 2023）

- **预训练 ckpt**：HumanML3D，50-step 快速版可用
- **架构**：Transformer encoder，$x_0$-prediction
- **关键帧能力**：模型本身未针对关键帧训练，推理时通过 imputation 或 test-time optimization 注入

### MoMask（CVPR 2024）

- **预训练 ckpt**：HumanML3D + KIT-ML，可下载
- **架构**：Masked Transformer + RVQ（非扩散模型）
- **关键帧能力**：将关键帧位置 token 保留，非关键帧设为 `[MASK]`，模型迭代填充。论文仅测试过连续区间 inpainting，稀疏关键帧设定为首次尝试

### CondMDI（SIGGRAPH 2024）

- **预训练 ckpt**：frame 插帧、frame-joint 插帧、uncond，均在 HumanML3D 上训练
- **架构**：Transformer encoder / U-Net，$\epsilon$-prediction
- **关键帧能力**：训练时 clean $x_0$ 替换 + mask concat（imputation ckpt），推理时同样用 imputation

---

## 约束方式

### 范式一：Imputation（硬替换）

采样时关键帧位置用 GT 加噪值硬替换

| 模型 | 说明 |
|------|------|
| MDM | 模型未针对 imputation 训练，预测和替换互相拉扯，缝合边界严重 |
| CondMDI | 模型针对 imputation 训练，预测和替换协调，缝合边界较轻 |

### 范式二：Test-time Optimization（噪声优化）

优化初始噪声 $z_T$ 使输出满足关键帧约束（DNO 方法），梯度通过整个采样链反传

| 模型 | 说明 |
|------|------|
| MDM | DNO 基于 MDM 构建，直接适用。软约束，无缝合边界，推理慢 |

### 范式三：Temporal Inpainting（mask 填充）

关键帧位置保留 token，非关键帧设为 `[MASK]`，模型迭代填充

| 模型 | 说明 |
|------|------|
| MoMask | 非扩散路线。无硬替换边界，但存在 RVQ 量化误差和 4 倍时间下采样精度限制 |

---

## 已知效果排序

**对未训练关键帧条件的模型（MDM）**：test-time optimization > imputation

**对已训练关键帧条件的模型（CondMDI）**：imputation 效果最优，不需要额外约束

**MoMask temporal inpainting 在稀疏关键帧设定下**：无已有数据，首次尝试

---

## 理论预估

| 模型 | 约束方式 | KF-MPJPE 预估 | Jitter 预估 | 核心 tradeoff |
|------|---------|-------------|------------|--------------|
| MDM | imputation | ≈ 0 | 最高 | 硬替换精确对齐，但模型未训练 imputation，缝合边界最严重 |
| MDM | test-time opt (DNO) | 中等 | 最低 | 软优化有残差，但全局一致无缝合边界 |
| MoMask | temporal inpainting | 中等偏高 | 低 | 量化误差 + 时间下采样限制对齐精度，但原生填充无边界 |
| CondMDI | imputation | ≈ 0 | 中等 | 硬替换精确对齐 + 模型已训练 imputation，综合最优 |

---

## 实验矩阵

| 模型 | 约束方式 | 约束范式 | KF-MPJPE | Jitter |
|------|---------|---------|----------|--------|
| MDM | imputation | 硬替换 | | |
| MDM | test-time opt (DNO) | 噪声优化 | | |
| MoMask | temporal inpainting | mask 填充 | | |
| CondMDI | imputation | 硬替换 | | |
