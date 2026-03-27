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

## 方案
