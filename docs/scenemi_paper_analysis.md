# SceneMI 论文逐节解剖

## 0. 一句话定位

将 HSI 建模问题**重新定义为 scene-aware motion in-betweening**，用扩散模型在 3D 场景约束下从稀疏关键帧生成完整动作序列，并利用扩散去噪特性天然处理带噪声的关键帧。

---

## 1. 问题定义（Section 3 开头）

### 输入
- 3D 场景 $\mathcal{G}$
- 稀疏关键帧 $\mathbf{s} \in \mathbb{R}^{N \times D}$，$N$ 为总帧数，$D=201$ 为每帧特征维度
- 关键帧 indicator $\mathbf{m} \in \{0,1\}^N$，$m_n=1$ 表示第 $n$ 帧是关键帧

### 输出
- 完整动作序列 $\mathbf{x} = \{x^n\} \in \mathbb{R}^{N \times D}$，同时满足关键帧约束和场景环境约束

### 动作表示（201 维）
| 成分 | 维度 | 说明 |
|------|------|------|
| Global joint positions $J$ | $22 \times 3 = 66$ | 包含 root translation $\gamma \in \mathbb{R}^3$ |
| 6D root orientation $\phi$ | $6$ | 连续 6D 旋转表示 |
| Local SMPL pose $\psi$ | $21 \times 6 = 126$ | 21 个非根关节的 6D 旋转 |
| 身体形状特征 $\mathbf{b}$ | $7$ | 代表性关节对之间的距离（如 root↔head, left_shoulder↔right_shoulder） |
| **总计** | **201 + 7（shape 单独输入）** | |

> **与 CondMDI 的区别**：CondMDI 使用 HumanML3D 的 263 维表示（相对根节点位移 + RIC 位置 + 连续旋转 + 局部速度 + 脚接触），SceneMI 使用**全局关节位置**，这对场景碰撞检测更友好（不需要先恢复全局坐标）。

---

## 2. 场景编码（Section 3.1）—— 你关注的核心

SceneMI 使用**双层场景描述符（Dual Scene Descriptors）**：

### 2.1 全局场景特征 $\mathbf{c}_g$

- **表示方式**：粗分辨率的 occupancy voxel grid，$\mathbf{c}_g \in \{0,1\}^{d_x \times d_y \times d_z}$
- **分辨率**：0.1m/voxel，网格大小 $48 \times 24 \times 48$（即覆盖 4.8m × 2.4m × 4.8m 的空间）
- **坐标系**：以第一帧的 root position 和 orientation 为中心和朝向初始化
- **编码器**：Vision Transformer (ViT)，输出 512 维特征向量
- **训练方式**：与整个框架端到端联合训练
- **CFG dropout**：训练时以 10% 概率 mask 掉 $\mathbf{c}_g$，支持推理时的 classifier-free guidance

**局限性分析**：
- 固定 4.8m × 2.4m × 4.8m 的范围，只能覆盖**中小型室内区域**
- 0.1m 分辨率对于精细几何（如椅子扶手、桌腿）表达力有限
- Occupancy（0/1 二值）丢失了表面法线、材质等信息
- 以第一帧为中心 → 不适合长距离位移的动作

### 2.2 局部场景特征 $\mathbf{c}_l$

- **表示方式**：Basis Point Set (BPS) 特征
- **构造过程**：
  1. 在 T-pose SMPL mesh 表面做 farthest-point sampling，选 64 个 anchor 点
  2. 这 64 个顶点索引在所有关键帧中固定不变
  3. 对每个关键帧，找场景中距离每个 anchor 最近的点
  4. 计算 offset 向量（anchor → 场景最近点），得到 $\mathbf{c}_l^n \in \mathbb{R}^{64 \times 3}$
- **编码器**：MLP，与关键帧特征在对应帧 concat

**BPS 的优势**：
- 与点的顺序、mesh 拓扑、分辨率无关 → 泛化性好
- 只关注身体附近的场景几何 → 聚焦交互相关区域
- 64 个点足以粗略描述"身体周围有没有东西"

**BPS 的局限**：
- 只在**关键帧位置**计算 → 非关键帧的局部场景信息靠模型自行推断
- 64 个 anchor 的分辨率较粗 → 可能遗漏细粒度接触（如手指抓握）
- 依赖 SMPL mesh 质量 → 对不同体型泛化性取决于 SMPL 参数化

### 2.3 场景表示是否是 SceneMI 的瓶颈？

**是的，场景表示是 SceneMI 泛化性的主要瓶颈之一。** 具体来说：

| 场景类型 | Occupancy Voxel 能否覆盖 | BPS 能否捕捉 | SceneMI 可行性 |
|---------|------------------------|------------|--------------|
| 小型室内（TRUMANS 风格） | 可以 | 可以 | 好 |
| 大型室内（开放大厅） | 4.8m 不够 | 关键帧附近可以 | 需扩展 voxel |
| 户外场景 | 完全不够 | 地面可以，远处不行 | 差 |
| 精细交互（桌面操作） | 0.1m 太粗 | 64 点太少 | 差 |
| 动态场景（移动家具） | 二值不变 | 关键帧时刻快照 | 不支持 |

---

## 3. 扩散模型架构（Section 3.2）

### 模型选择
- **U-Net** + AdaGN (Adaptive Group Normalization) + 1D 卷积
- 不是 CondMDI 的 Transformer encoder，而是类似 GMD 的 U-Net
- AdaGN 动态调整归一化，编码 diffusion timestep $t$ 的信息

### 条件注入方式
- **Motion features**：$\tilde{\mathbf{x}}_t = \text{spatialconcat}(\mathbf{x}_t', \mathbf{c}_l', \mathbf{m})$
  - $\mathbf{x}_t'$：imputed 后的 noisy motion（关键帧位置用 clean 值替换）
  - $\mathbf{c}_l'$：imputed 后的局部场景特征（只保留关键帧位置的 BPS）
  - $\mathbf{m}$：关键帧 indicator mask
- **Global features**：$\mathbf{c}_g$（全局场景）+ $\mathbf{b}$（身体形状）沿时间维度 concat
- **Timestep embedding**：加到所有输入特征上

### 训练 Loss

$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_J \mathcal{L}_{\text{joints}} + \lambda_V \mathcal{L}_{\text{velocity}}$$

| Loss | 公式 | 作用 |
|------|------|------|
| $\mathcal{L}_{\text{simple}}$ | $\|\mathbf{x}_0 - \mathcal{D}_\theta(\mathbf{x}_t, t, \tau)\|_2^2$ | 基础重建 loss（预测 $x_0$） |
| $\mathcal{L}_{\text{joints}}$ | $\|\text{FK}(\mathbf{x}_0) - \text{FK}(\hat{\mathbf{x}}_0)\|_2^2$ | FK 后的关节位置 loss，物理合理性 |
| $\mathcal{L}_{\text{velocity}}$ | $\|\dot{\mathbf{x}}_0 - \dot{\hat{\mathbf{x}}}_0\|_2^2$ | 速度 loss，平滑性 |

> **注意**：与 CondMDI 不同，SceneMI 预测 $x_0$（`ModelMeanType.START_X`），不是预测 $\epsilon$。

---

## 4. 关键帧处理机制（Section 3.2，核心创新）

### 4.1 训练时的 Imputation

每个训练 step：
1. 随机采样 diffusion timestep $t \sim \mathcal{U}(\{1,...,T\})$
2. 随机选 $k \sim \mathcal{U}(\{2,...,N\})$ 个关键帧（**必含首尾帧**）
3. Imputation：$\mathbf{x}_t' = \mathbf{m} \odot \mathbf{x}_0 + (1 - \mathbf{m}) \odot \mathbf{x}_t$
   - 关键帧位置：替换为 **clean** $\mathbf{x}_0$（不加噪）
   - 非关键帧位置：保持 noisy $\mathbf{x}_t$

> **与 CondMDI 的区别**：CondMDI 的 imputation 是在关键帧位置替换为 $q(x_t | x_0)$（加了噪的 GT），SceneMI 直接用 clean $x_0$。这使得模型学会"关键帧位置的值是干净的参考信号"。

### 4.2 推理时的两阶段去噪（T* 机制）

这是处理噪声关键帧的核心设计：

**阶段 1：$t = T \to T^*+1$**
- 正常 imputation：关键帧位置用（可能带噪的）输入关键帧替换
- 模型利用带噪关键帧建立全局结构

**阶段 2：$t = T^* \to 1$**
- **停止 imputation**，让模型对整个序列（包括关键帧位置）自主去噪
- 扩散模型的去噪能力自然修正关键帧中的噪声

$$T^* = 20 \text{（论文中的超参选择，总步数 } T=1000 \text{）}$$

**为什么 $T^*=20$ 有效**：
- 在 $t=20$ 时，$\bar{\alpha}_{20} \approx 0.97$，信号已经 97% 确定
- 前 980 步已经建立了正确的全局结构
- 最后 20 步负责高频细节平滑——正是 jitter 和噪声被修正的阶段
- 如果 $T^*$ 太大（如 100），放开太早，结构可能漂移
- 如果 $T^*$ 太小（如 5），放开太晚，不够时间平滑

### 4.3 噪声关键帧训练增强（Section 3.2.1）

为了让模型在推理时能处理噪声关键帧，训练时增加数据增强：

- 对 clean 关键帧添加合成噪声
- 噪声类型模拟真实传感器误差（IMU 漂移、深度估计误差等）
- 具体细节在补充材料中

> **这是你可以直接借鉴的技术**：在你的 CondMDI 训练中对 GT 关键帧添加噪声，模拟粗提取的误差。

---

## 5. 实验设计（Section 4）

### 5.1 数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| TRUMANS | 训练 + 测试 | 15+ 小时 MoCap，100 个手工室内场景，clean |
| GIMO | 仅测试（零样本泛化） | IMU 传感器 + 手机扫描，noisy |
| PROX | 仅测试（视频重建） | 单目视频 + RGB 估计的 pose |

### 5.2 评估指标

| 指标 | 说明 |
|------|------|
| **FID** | 生成运动的分布质量 |
| **Diversity** | 生成多样性 |
| **Foot Skating** | 脚部滑动量 |
| **Jitter** | 动作抖动（关节加速度的大小） |
| **Penetration** | 人体与场景的穿透 |
| **Contact** | 接触合理性 |

### 5.3 关键实验结果

**Table 1: Clean Keyframes on TRUMANS**
- SceneMI 在所有场景感知指标上优于 MDM 和 CondMDI
- 相比 "Without Scene-awareness" 变体：Penetration 显著降低

**Table 3: Noisy Keyframes on TRUMANS**
- 加入 $T^*$ 机制后 jitter 和 foot skating 大幅降低
- 对比不用 $T^*$（全程 impute）：模型会"忠实复制"关键帧的噪声

**Table 4: Generalization to GIMO**
- **Jitter 降 56.5%**（0.573 → 0.249）
- **Foot skating 降 37.5%**（0.261 → 0.163）
- 零样本泛化到完全不同的场景来源和动作质量

### 5.4 消融实验

**场景编码消融**：
- 去掉全局场景 → Penetration 显著上升
- 去掉局部 BPS → 精细交互质量下降
- 两者缺一不可

**$T^*$ 消融**：
- $T^*=0$（全程 impute，不放开）：noisy 关键帧的噪声直接传到输出
- $T^*=T$（全程不 impute）：失去关键帧约束
- $T^*=20$：最佳平衡点

---

## 6. 视频重建应用（Section 4.3）

**这不是 SceneMI 的核心贡献，只是一个应用展示。**

Pipeline（拼接多个现有方法）：
1. 实例分割 [SAM]
2. Image-to-3D 重建 [Wonder3D, InstantMesh]
3. 深度估计 [DepthAnything] + 相机参数估计 [DUSt3R]
4. Human mesh recovery [SMPLer-X]
5. **SceneMI 做 in-betweening 精修**（关键帧间隔 15 帧）
6. 自回归采样延长序列（用最后 60 帧作为下一段的初始关键帧）

---

## 7. 局限性总结

| 局限 | 具体表现 | 影响范围 |
|------|---------|---------|
| **场景表示受限** | 4.8m voxel + 64 点 BPS | 大场景、户外、精细交互 |
| **训练数据单一** | 仅 TRUMANS | 动作类型和场景类型有限 |
| **不支持动态场景** | Occupancy 是静态的 | 移动物体、开关门等 |
| **不支持物体交互** | 只建模人-静态场景 | 抓取、推拉等 |
| **全局坐标依赖** | 以第一帧为原点 | 长距离位移场景 |
| **U-Net 架构** | 1D 卷积 + AdaGN | 相比 Transformer 可能在长序列上弱 |

---

## 8. 对你工作的启示

### 可直接借鉴的
1. **$T^*$ 两阶段机制**：你的 CondMDI 代码库已有 `impute_until` + `second_stage` 支持
2. **训练时关键帧加噪**：模拟粗提取误差
3. **首尾帧必含**：SceneMI 训练时随机选关键帧但强制包含首尾帧
4. **Joint position + velocity 联合 loss**：提升平滑度

### 你的工作相比 SceneMI 的优势
1. **不依赖特定场景表示**：你的约束策略（pelvis dense + sparse keyframe）纯运动学，不需要 occupancy voxel 或 BPS
2. **更灵活的关节级控制**：SceneMI 只做帧级 imputation（整帧锁或整帧放），你可以做关节级（碰撞关节锁、非碰撞关节放）
3. **更通用的训练数据**：HumanML3D 比 TRUMANS 动作类型更丰富
4. **与 ZeroHSI 等上游更兼容**：不需要场景编码转换

### SceneMI 的场景编码如果要引入你的工作
- **全局场景**：可以考虑轻量替代（BEV heightmap、SDF 等），不一定要 occupancy voxel
- **局部场景**：BPS 思路值得借鉴——只编码身体附近的场景几何
- **关键问题**：你的场景数据是什么格式？这决定了该用哪种场景表示
