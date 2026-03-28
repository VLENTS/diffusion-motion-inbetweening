# 缝合边界问题的数学分析与修复方案

## 问题定义

在使用 imputation 做关键帧约束的扩散采样中，最终输出在关键帧与非关键帧的边界处出现不连续（高 Jitter），称为缝合边界问题

---

## 我们想要什么

从条件分布中采样：

$$p(\mathbf{x}_0^{\text{free}} \mid \mathbf{x}_0^{\text{kf}} = \mathbf{x}_0^{gt})$$

即在关键帧位置等于 GT 的条件下，非关键帧位置的分布。从这个分布采出来的样本**天然**和关键帧是协调的，不存在缝合边界

---

## Imputation 实际做了什么

在每步 $t$，imputation 把采样分成两条**独立的路径**：

关键帧位置：

$$x_{t-1}^{\text{kf}} = \sqrt{\bar{\alpha}_{t-1}}\, x_0^{gt} + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_1$$

非关键帧位置：

$$x_{t-1}^{\text{free}} = \mu_\theta(x_t, t) + \sigma_t\, \epsilon_2$$

其中 $\epsilon_1$ 和 $\epsilon_2$ 是**独立的**随机噪声

在正确的反向过程中，$x_{t-1}$ 的所有位置应该来自**同一个**后验分布 $p_\theta(x_{t-1} \mid x_t)$，位置之间的相关性由模型的全局预测 $\hat{x}_0$ 维持。Imputation 在关键帧位置用独立噪声 $\epsilon_1$ 替换了模型的预测，**破坏了关键帧和非关键帧之间的相关结构**

---

## 不一致在哪里积累

第 $t$ 步，模型接收到 $x_t$（关键帧位置已被 impute）。关键帧位置的值包含 imputation 噪声 $\epsilon_t^{(1)}$，非关键帧位置的值来自上一步模型的采样。模型的 Transformer/U-Net 联合处理整个序列，它的预测 $\hat{x}_0[i\!+\!1]$（非关键帧）受 $x_t[i]$（关键帧，含独立噪声）影响

模型在训练时看到的输入是：关键帧位置 = clean $x_0$，非关键帧 = 噪声 $x_t$。但推理时 imputation 放入的是 $q(x_t \mid x_0^{gt})$（加了噪声的 GT），不是 clean $x_0$。这是另一层不匹配

每一步都引入独立噪声 → 每一步模型看到的关键帧信号都和它"期望"的不完全一致 → 模型在非关键帧位置的预测有微小偏差 → 偏差在 1000 步中累积 → 最终输出的关键帧位置（= GT）和非关键帧位置（= 累积偏差后的模型预测）之间出现不连续

---

## Jitter 在边界处的公式

设帧 $i$ 是关键帧，帧 $i\!-\!1, i\!+\!1, i\!+\!2$ 是非关键帧。最终输出中：

$$x_0[i] = x_0^{gt}[i] \quad \text{（impute 硬替换）}$$

$$x_0[i\!+\!1] = \hat{x}_0^{\text{model}}[i\!+\!1] \quad \text{（模型预测，受累积偏差影响）}$$

边界处的 Jitter 贡献：

$$J_{\text{boundary}} = \left\lVert x_0[i\!+\!2] - 3\,x_0[i\!+\!1] + 3\,\underbrace{x_0^{gt}[i]}_{\text{imputed}} - x_0[i\!-\!1] \right\rVert$$

四个值中 $x_0[i]$ 来自 GT，其余三个来自模型。如果模型预测与 GT 完全一致则 $J = J_{gt}$。但因为累积偏差，$x_0[i\!-\!1], x_0[i\!+\!1], x_0[i\!+\!2]$ 偏离了 GT → $J_{\text{boundary}} > J_{gt}$

---

## 根因

根因不是 Jitter 本身——Jitter 是症状。根因是 **imputation 不是从正确的条件分布采样的**

正确的采样应该是：找到一个 $x_0 \sim p(x_0)$，使得 $x_0^{\text{kf}} = x_0^{gt}$。这等价于从 $p(x_0 \mid x_0^{\text{kf}} = x_0^{gt})$ 中采样

Imputation 试图通过每步局部替换来近似这个条件采样，但独立噪声破坏了全局一致性

---

## 修复方案

### 方案 A：Imputation + DNO Jitter Loss

在 imputation 采样的基础上，用 DNO 方法优化初始噪声 $z_T$，使经过 imputation 采样后的输出 Jitter 最小：

$$z_T^* = \arg\min_{z_T} \text{Jitter}\!\left(\text{SamplingChain}_{\text{with imputation}}(d(\cdot), z_T)\right)$$

数学含义：在 imputation 这个"有缺陷的采样过程"中，找到一个让缺陷影响最小的起点。不消除 imputation 的根本问题（独立噪声注入），而是找到一个 $z_T$ 使得模型的去噪轨迹和 imputation 的独立噪声"对齐"

| KF-MPJPE | Jitter |
|----------|--------|
| ≈ 0（imputation 保证） | 降低（$z_T$ 优化） |

### 方案 B：纯 DNO（KF-MPJPE + Jitter 联合优化）

不做任何 imputation，纯粹优化 $z_T$：

$$z_T^* = \arg\min_{z_T} \left[\lambda_1 \cdot \text{KF-MPJPE}(x_0, x_0^{gt}) + \lambda_2 \cdot \text{Jitter}(x_0)\right]$$

其中 $x_0 = \text{ODESolver}(d(\cdot), z_T)$

数学含义：直接从 $p(x_0)$ 中找满足两个目标的样本。没有独立噪声注入，没有缝合边界。整个采样链是一条连贯的 ODE 轨迹，所有位置的相关结构完全由模型维持

| KF-MPJPE | Jitter |
|----------|--------|
| > 0（软优化，有残差） | 最低（无缝合边界） |

### 方案 C：Imputation + DNO（KF-MPJPE + Jitter 联合优化）

结合 A 和 B：imputation 保证关键帧对齐，DNO 同时优化关键帧匹配和 Jitter：

$$z_T^* = \arg\min_{z_T} \left[\lambda_1 \cdot \text{KF-MPJPE}(x_0, x_0^{gt}) + \lambda_2 \cdot \text{Jitter}(x_0)\right]$$

其中 $x_0 = \text{SamplingChain}_{\text{with imputation}}(d(\cdot), z_T)$

这里 KF-MPJPE 项是冗余的（imputation 已经保证了），但加上它可以让 $z_T$ 的优化方向和 imputation 的替换方向一致，减少两者之间的冲突

| KF-MPJPE | Jitter |
|----------|--------|
| ≈ 0（imputation 保证 + DNO 协同） | 降低 |

---

## 方案对比

| 方案 | 采样方式 | KF-MPJPE | Jitter | 数学性质 |
|------|---------|----------|--------|---------|
| 纯 Imputation（当前） | 每步独立噪声替换 | = 0 | 高 | 破坏相关结构 |
| A: Imputation + DNO(Jitter) | 优化 $z_T$ + 每步替换 | ≈ 0 | 降低 | 缺陷仍在，但起点最优 |
| B: 纯 DNO(KF+Jitter) | 优化 $z_T$，不替换 | > 0 | 最低 | 从正确分布采样的近似 |
| C: Imputation + DNO(KF+Jitter) | 优化 $z_T$ + 每步替换 | ≈ 0 | 降低 | A 的强化版 |

选择取决于对 KF-MPJPE 的容忍度：
- 必须精确为零 → 方案 A 或 C
- 允许小残差 → 方案 B 最干净

---

## 可能的 Jitter Loss 定义

### 定义 1：直接 Jitter

$$\mathcal{L}_{\text{jitter}} = \frac{1}{(T\!-\!3) \cdot J} \sum_{t=1}^{T-3} \sum_{j=1}^{J} \frac{\left\lVert \mathbf{x}_{t+3,j} - 3\,\mathbf{x}_{t+2,j} + 3\,\mathbf{x}_{t+1,j} - \mathbf{x}_{t,j} \right\rVert_2}{\Delta t^3}$$

直接最小化三阶差分。最直接地针对 Jitter 症状

### 定义 2：速度平滑

$$\mathcal{L}_{\text{vel}} = \frac{1}{(T\!-\!2) \cdot J} \sum_{t=1}^{T-2} \sum_{j=1}^{J} \left\lVert (\mathbf{x}_{t+2,j} - \mathbf{x}_{t+1,j}) - (\mathbf{x}_{t+1,j} - \mathbf{x}_{t,j}) \right\rVert_2$$

最小化加速度变化（二阶差分）。比 Jitter 低一阶，约束更平滑但可能不够直接

### 定义 3：边界局部 Jitter

$$\mathcal{L}_{\text{boundary}} = \sum_{i \in \mathcal{K}} \sum_{j=1}^{J} \left\lVert \mathbf{x}_{i+2,j} - 3\,\mathbf{x}_{i+1,j} + 3\,\mathbf{x}_{i,j} - \mathbf{x}_{i-1,j} \right\rVert_2$$

只在关键帧边界（$\pm 2$ 帧范围内）计算 Jitter。比全局 Jitter 更精准地针对缝合问题，计算量更小，梯度更集中
