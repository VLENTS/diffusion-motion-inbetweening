# 两种 Imputation 训练策略的本质区别与修正能力分析

> **勘误**：CondMDI 训练时关键帧位置放的也是 clean $\mathbf{x}_0$（见 `mdm_unet.py` 第 781 行：`x = obs_x0 * obs_mask + x * (~obs_mask)`），不是 noised imputation。两者训练时的 imputation 值相同，差异在于模型架构（$\epsilon$-prediction vs $x_0$-prediction）和 $T^*$ 放开时的 mask 处理方式。

## 1. 训练时两种策略在做什么

设总 diffusion 步数为 $T=1000$，当前时间步为 $t$，clean 动作为 $\mathbf{x}_0$，标准扩散前向过程为：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

关键帧 mask 为 $\mathbf{m} \in \{0,1\}^N$，$m_n=1$ 表示第 $n$ 帧是关键帧。

### CondMDI：Noised Imputation

在关键帧位置替换为**与当前 $t$ 匹配的加噪 GT**：

$$\mathbf{x}_t^{\text{input}} = \mathbf{m} \odot \underbrace{\left(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}\right)}_{\text{关键帧：加了 } t \text{ 级噪声的 GT}} + (1-\mathbf{m}) \odot \underbrace{\mathbf{x}_t}_{\text{非关键帧：正常 } x_t}$$

在关键帧位置，输入的噪声水平 = $t$。
在非关键帧位置，输入的噪声水平 = $t$。
**整个输入的噪声水平是均匀的。**

### SceneMI：Clean Imputation

在关键帧位置替换为**干净的 GT**（不加噪）：

$$\mathbf{x}_t^{\text{input}} = \mathbf{m} \odot \underbrace{\mathbf{x}_0}_{\text{关键帧：clean}} + (1-\mathbf{m}) \odot \underbrace{\mathbf{x}_t}_{\text{非关键帧：正常 } x_t}$$

在关键帧位置，输入的噪声水平 = $0$。
在非关键帧位置，输入的噪声水平 = $t$。
**输入中存在噪声水平的不对称。**

### 本质区别

两种策略的输入只差一项，就是关键帧位置上的值：

$$\Delta = \mathbf{m} \odot \left(\mathbf{x}_0 - \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}\right) = \mathbf{m} \odot \left((1-\sqrt{\bar{\alpha}_t})\,\mathbf{x}_0 - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}\right)$$

当 $t$ 大时（如 $t=500$，$\sqrt{\bar{\alpha}_t} \approx 0.5$），$\Delta$ 很大——两种策略的输入差距巨大。
当 $t$ 小时（如 $t=20$，$\sqrt{\bar{\alpha}_t} \approx 0.985$），$\Delta$ 很小——两种策略的输入几乎相同。

**本质**：两种策略定义了模型对关键帧位置的不同"角色认知"：

| | 关键帧位置的角色 | 模型学到的行为 |
|---|---|---|
| CondMDI | 与其他位置一样是待去噪的信号 | "我需要从噪声中恢复 $x_0$，关键帧位置要更忠实" |
| SceneMI | 干净的参考信号 | "关键帧位置是真值，我要利用它去生成其余帧" |

---

## 2. 为什么 CondMDI 的 $T^*$ 修正能力弱

### 去噪过程的数学

模型预测 $\hat{\mathbf{x}}_0$ 后，单步去噪（DDPM）计算 $\mathbf{x}_{t-1}$ 的过程为：

$$\mathbf{x}_{t-1} = \underbrace{\frac{\sqrt{\bar{\alpha}_{t-1}}\,(1-\alpha_t)}{1-\bar{\alpha}_t}\,\hat{\mathbf{x}}_0 + \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t}_{\text{posterior mean } \mu_\theta(\mathbf{x}_t, t)} + \sigma_t \mathbf{z}$$

其中 $\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t, \text{conditions})$ 是模型的预测。

每步去噪对当前值 $\mathbf{x}_t$ 的修正量为：

$$\delta_t = \mathbf{x}_{t-1} - \mathbf{x}_t$$

这个修正量的大小由两个因素决定：
- $\hat{\mathbf{x}}_0$ 与 $\mathbf{x}_t$ 的差距
- 系数 $\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t}$，这个系数随 $t$ 减小而减小

### 在 $t=20$ 时的修正能力

在 $t=20$ 时，$\bar{\alpha}_{20} \approx 0.97$，所以：

$$\mathbf{x}_{20} \approx 0.985\,\mathbf{x}_0^{\text{true}} + 0.17\,\boldsymbol{\epsilon}$$

此时 $\mathbf{x}_{20}$ 已经非常接近 clean 数据。模型的预测 $\hat{\mathbf{x}}_0$ 也会非常接近 $\mathbf{x}_{20}$。

**CondMDI 的情况**：

模型是 $\epsilon$-prediction（预测噪声），预测的 clean 数据为：

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\boldsymbol{\epsilon}}_\theta}{\sqrt{\bar{\alpha}_t}}$$

在 $t=20$，代入数值：

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_{20} - 0.17\,\hat{\boldsymbol{\epsilon}}_\theta}{0.985}$$

修正量 $\hat{\mathbf{x}}_0 - \mathbf{x}_{20}$ 的量级为：

$$\hat{\mathbf{x}}_0 - \mathbf{x}_{20} \approx \frac{0.17\,(\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta)}{0.985} + \frac{0.015}{0.985}\,\mathbf{x}_{20}$$

关键问题在于：如果关键帧位置上存在来自传感器的噪声 $\boldsymbol{\eta}$（不是扩散噪声 $\boldsymbol{\epsilon}$），则实际输入为：

$$\mathbf{x}_{20}^{\text{keyframe}} \approx 0.985\,(\mathbf{x}_0 + \boldsymbol{\eta}) + 0.17\,\boldsymbol{\epsilon}$$

模型的 $\hat{\boldsymbol{\epsilon}}_\theta$ 只被训练来预测扩散噪声 $\boldsymbol{\epsilon}$，对 $\boldsymbol{\eta}$ 是无感的。所以：

$$\hat{\mathbf{x}}_0^{\text{keyframe}} \approx \mathbf{x}_0 + \boldsymbol{\eta} + \text{small correction from } \boldsymbol{\epsilon}$$

**传感器噪声 $\boldsymbol{\eta}$ 被完整保留。** 模型减去的是它对扩散噪声 $\boldsymbol{\epsilon}$ 的估计，不会去碰 $\boldsymbol{\eta}$。

**SceneMI 的情况**：

模型是 $x_0$-prediction（直接预测 clean 数据）：

$$\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t, \text{conditions})$$

模型在训练时，关键帧位置的输入永远是 clean $\mathbf{x}_0$，loss 驱动模型在这些位置输出 clean 值。模型学到的函数 $f_\theta$ 在关键帧位置的行为是：

$$f_\theta(\cdot)\big|_{\text{keyframe}} \approx \text{projection onto clean motion manifold}$$

这个投影不通过"估计扩散噪声量"来实现，而是直接由运动先验驱动——输出应该像自然动作。

当输入包含传感器噪声 $\boldsymbol{\eta}$ 时：

$$\hat{\mathbf{x}}_0^{\text{keyframe}} = f_\theta(\mathbf{x}_0 + \boldsymbol{\eta}, \ldots) \approx \text{proj}_{\text{manifold}}(\mathbf{x}_0 + \boldsymbol{\eta})$$

如果 $\boldsymbol{\eta}$ 不大，这个投影会把 $\mathbf{x}_0 + \boldsymbol{\eta}$ 拉回流形，**$\boldsymbol{\eta}$ 被压缩**。

### 修正能力对比总结

$$\boxed{\text{CondMDI}: \quad \hat{\mathbf{x}}_0 \approx \mathbf{x}_0 + \boldsymbol{\eta} \quad (\text{传感器噪声保留})}$$

$$\boxed{\text{SceneMI}: \quad \hat{\mathbf{x}}_0 \approx \text{proj}_{\mathcal{M}}(\mathbf{x}_0 + \boldsymbol{\eta}) \quad (\text{传感器噪声被投影压缩})}$$

其中 $\mathcal{M}$ 是模型学到的自然动作流形。

### 直觉理解

| | CondMDI ($\epsilon$-prediction + noised imputation) | SceneMI ($x_0$-prediction + clean imputation) |
|---|---|---|
| 模型的工作 | "估计并减去扩散噪声" | "预测干净数据应该长什么样" |
| 对扩散噪声 $\epsilon$ | 能去除 | 能去除 |
| 对传感器噪声 $\eta$ | **不去除**（不在训练分布内） | **能压缩**（投影到流形） |
| 修正力度取决于 | timestep $t$（$t$ 小 → 修正小） | 运动先验（与 $t$ 无关） |
