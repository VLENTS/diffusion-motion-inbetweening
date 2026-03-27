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

### 模型选择依据

| 角色 | 选择 | 理由 |
|------|------|------|
| 里程碑模型 | **MDM**（ICLR 2023） | 奠定了运动扩散模型的基础范式（Transformer + diffusion + CLIP），后续 CondMDI/GMD/OmniControl/DNO 全部基于它构建 |
| 动作质量 SOTA | **MoMask**（CVPR 2024） | HumanML3D 上 FID=0.045（同期最优），非扩散路线（Masked Transformer + RVQ），提供不同技术路线的对照 |
| 关键帧可控 SOTA | **CondMDI**（SIGGRAPH 2024） | 唯一一个在训练时就学习关键帧条件的模型，支持任意帧数/任意关节子集的灵活约束，关键帧遵循度最高 |

### 模型详情

#### MDM（ICLR 2023）

- **仓库**：[GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)
- **预训练 ckpt**：HumanML3D，50-step 快速版可用
- **架构**：Transformer encoder，$x_0$-prediction
- **关键帧能力**：推理时通过 inpainting 注入，模型本身未针对关键帧训练

#### MoMask（CVPR 2024）

- **仓库**：[EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes)
- **预训练 ckpt**：HumanML3D + KIT-ML，可下载
- **架构**：Masked Transformer + RVQ（非扩散模型）
- **关键帧能力**：temporal inpainting，mask 掉非关键帧区域让模型填充

#### CondMDI（SIGGRAPH 2024）

- **仓库**：[setarehc/diffusion-motion-inbetweening](https://github.com/setarehc/diffusion-motion-inbetweening)
- **预训练 ckpt**：frame 插帧、frame-joint 插帧、uncond，均在 HumanML3D 上训练
- **架构**：Transformer encoder / U-Net，$\epsilon$-prediction
- **关键帧能力**：训练时 clean $x_0$ 替换 + mask concat，原生支持关键帧条件

---

## 约束方式（按范式整理）

### 范式一：硬替换（Imputation）

在采样的每一步（或部分步），将关键帧位置的值强制替换为 GT 派生的值

| 模型 | 约束方式 | 说明 |
|------|---------|------|
| MDM | inpainting (RePaint) | 每步在关键帧位置用 $q(x_{t-1} \mid x_0^{gt})$ 替换采样值 |
| CondMDI | impute (stop=0) | 全程 imputation，关键帧每步被 GT 加噪值替换 |
| CondMDI | impute (stop=1) | 最后一步不 impute，模型自主平滑最终输出 |
| CondMDI | dual-phase (t\*=20) | t≥20 密锁，t<20 换稀疏 mask（减少缝合边界） |

**特点**：KF-MPJPE 最低（硬约束保证对齐），但产生缝合边界影响 Jitter

### 范式二：梯度引导（Guidance）

不替换采样值，通过对关键帧约束求梯度来引导去噪方向

| 模型 | 约束方式 | 说明 |
|------|---------|------|
| CondMDI | reconstruction guidance | 对 $\lVert x_0^{gt} - \hat{x}_0 \rVert^2$ 在关键帧位置求梯度，引导预测 |
| CondMDI | impute + recg | 硬替换 + 梯度引导叠加 |

**特点**：软约束，不产生缝合边界，Jitter 更优，但 KF-MPJPE 可能偏高

### 范式三：Mask 填充（非扩散）

将非关键帧位置 mask 掉，模型直接生成填充

| 模型 | 约束方式 | 说明 |
|------|---------|------|
| MoMask | temporal inpainting | mask 非关键帧的 token，Masked Transformer 迭代填充 |

**特点**：非扩散路线的对照，无 imputation/guidance 机制，质量取决于 masked modeling 的能力

---

## 实验矩阵

| 模型 | 约束方式 | 约束范式 | KF-MPJPE | Jitter |
|------|---------|---------|----------|--------|
| MDM | inpainting (RePaint) | 硬替换 | | |
| MoMask | temporal inpainting | mask 填充 | | |
| CondMDI | impute (stop=0) | 硬替换 | | |
| CondMDI | impute (stop=1) | 硬替换 | | |
| CondMDI | reconstruction guidance | 梯度引导 | | |
| CondMDI | impute + recg | 硬替换 + 梯度引导 | | |
| CondMDI | dual-phase (t\*=20) | 硬替换（两阶段） | | |
