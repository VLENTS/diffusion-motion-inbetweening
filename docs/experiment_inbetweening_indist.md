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

## 可用模型与约束方式

### 1. CondMDI（SIGGRAPH 2024）

- **仓库**：[setarehc/diffusion-motion-inbetweening](https://github.com/setarehc/diffusion-motion-inbetweening)
- **预训练 ckpt**：frame 插帧、frame-joint 插帧、uncond，均在 HumanML3D 上训练
- **架构**：Transformer encoder / U-Net，$\epsilon$-prediction
- **关键帧注入方式**：训练时 clean $x_0$ 替换 + mask concat

| 约束方式 | 说明 | 预期特点 |
|---------|------|---------|
| impute (stop=0) | 全程 imputation，关键帧每步被 GT 加噪值替换 | KF-MPJPE 最低，Jitter 受缝合边界影响 |
| impute (stop=1) | 最后一步不 impute，模型自主平滑 | Jitter 降低，KF-MPJPE 略升 |
| impute + recg | imputation + reconstruction guidance 叠加 | 软硬约束结合 |
| dual-phase (stop=20) | t≥20 密锁，t<20 换稀疏 mask | Jitter 进一步降低，KF-MPJPE 取决于稀疏 mask 设计 |

### 2. OmniControl（ICLR 2024）

- **仓库**：[neu-vi/OmniControl](https://github.com/neu-vi/OmniControl)
- **预训练 ckpt**：HumanML3D 上训练，可下载
- **架构**：基于 MDM，ControlNet 风格的 copy branch + analytic guidance
- **关键帧注入方式**：推理时 analytic spatial guidance + realism guidance，不修改模型输入

| 约束方式 | 说明 | 预期特点 |
|---------|------|---------|
| spatial guidance | 通过梯度引导关节位置匹配目标 | 软约束，不产生缝合边界，Jitter 可能更低 |
| spatial + realism guidance | 叠加全关节的 realism 引导 | 平衡约束遵循度和动作自然度 |

### 3. DNO（CVPR 2024）

- **仓库**：[korrawe/Diffusion-Noise-Optimization](https://github.com/korrawe/Diffusion-Noise-Optimization)
- **预训练 ckpt**：基于 MDM + EMA，HumanML3D 上训练
- **架构**：不改模型，优化 diffusion 初始噪声 $z_T$
- **关键帧注入方式**：test-time optimization，将关键帧约束作为 loss 反传到 $z_T$

| 约束方式 | 说明 | 预期特点 |
|---------|------|---------|
| noise optimization (keyframe loss) | 优化 $z_T$ 使输出在关键帧处匹配 GT | 无缝合边界，全局一致，但推理慢 |
| noise optimization (keyframe + velocity loss) | 加入速度平滑正则 | 进一步降低 Jitter |

### 4. GMD（ICCV 2023）

- **仓库**：[korrawe/guided-motion-diffusion](https://github.com/korrawe/guided-motion-diffusion)
- **预训练 ckpt**：HumanML3D 上训练，可下载
- **架构**：U-Net，$x_0$-prediction
- **关键帧注入方式**：推理时 imputation + reconstruction guidance

| 约束方式 | 说明 | 预期特点 |
|---------|------|---------|
| imputation | 关键帧位置硬替换 | 与 CondMDI 类似 |
| reconstruction guidance | 梯度引导 | 软约束 |
| imputation + guidance | 两者叠加 | 与 CondMDI impute+recg 对比 |

### 5. PriorMDM（ICLR 2024）

- **仓库**：[priorMDM/priorMDM](https://github.com/priorMDM/priorMDM)
- **预训练 ckpt**：基于 MDM，HumanML3D 上训练，提供 DiffusionBlending 的微调模型
- **架构**：MDM 作为先验，DiffusionBlending 融合多个微调模型
- **关键帧注入方式**：DiffusionBlending 做关节级/轨迹级控制

| 约束方式 | 说明 | 预期特点 |
|---------|------|---------|
| DiffusionBlending | 融合关节控制模型的预测 | 关节级精细控制 |

### 6. MoMask（CVPR 2024）

- **仓库**：[EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes)
- **预训练 ckpt**：HumanML3D + KIT-ML，可下载
- **架构**：Masked Transformer + RVQ（非扩散模型）
- **关键帧注入方式**：temporal inpainting，mask 掉非关键帧让模型填充

| 约束方式 | 说明 | 预期特点 |
|---------|------|---------|
| temporal inpainting | mask 非关键帧区域，模型生成填充 | 非扩散路线的对照组 |

### 7. MDM（ICLR 2023）

- **仓库**：[GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)
- **预训练 ckpt**：HumanML3D，50-step 快速版可用
- **架构**：Transformer encoder，$x_0$-prediction
- **关键帧注入方式**：推理时 inpainting（RePaint 风格）

| 约束方式 | 说明 | 预期特点 |
|---------|------|---------|
| inpainting (RePaint) | 每步在关键帧位置用 GT 加噪替换 | 基础 baseline |

---

## 推荐实验矩阵

按优先级排序，覆盖三种约束范式（硬替换 / 梯度引导 / 噪声优化）：

| 模型 | 约束方式 | 约束范式 | KF-MPJPE | Jitter |
|------|---------|---------|----------|--------|
| CondMDI | impute (stop=0) | 硬替换 | | |
| CondMDI | impute (stop=1) | 硬替换 | | |
| CondMDI | dual-phase (stop=20) | 硬替换（两阶段） | | |
| OmniControl | spatial + realism guidance | 梯度引导 | | |
| DNO | noise optimization | 噪声优化 | | |
| GMD | imputation + guidance | 硬替换 + 梯度引导 | | |
| MoMask | temporal inpainting | mask 填充 | | |
| MDM | inpainting (RePaint) | 硬替换 | | |
