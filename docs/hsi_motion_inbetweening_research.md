# HSI 粗提取人体序列的 Motion In-betweening 研究综述与方案建议

## 1. 问题定义

**目标**: 使用粗提取的 HSI（Human-Scene Interaction）人体序列作为关键帧/稀疏约束，通过 motion in-betweening 生成符合 GT 动作、质量更高的完整动作序列。

**核心挑战**:
- 粗提取的 HSI 序列存在噪声（penetration、floating、skating 等物理不合理性）
- 关键帧本身可能不精确，直接作为硬约束会传播误差
- 需要在保持场景交互合理性的同时生成自然流畅的过渡动作
- 生成动作需尽量贴合 GT 分布

---

## 2. 当前代码库分析（CondMDI）

本仓库实现了 **CondMDI (Conditional Motion Diffusion In-betweening)**（SIGGRAPH 2024），核心机制：

### 2.1 关键帧条件注入方式
- **训练时**: 随机采样关键帧和关节子集，将其 concat 到 noisy input 上作为条件（`model/mdm.py` 中的 `CondProcess`）
- **推理时**: 两种策略
  - **Imputation（替换）**: 在每步去噪时，将已知关键帧区域的噪声数据替换为从 GT 加噪得到的值（`diffusion/gaussian_diffusion.py`）
  - **Reconstruction Guidance（重建引导）**: 对去噪输出与 GT 关键帧之间的 MSE loss 求梯度，引导生成向关键帧靠近（`utils/editing_util.py` 中的 `grad_fn`）

### 2.2 当前局限
- 假设关键帧是 **精确的**（直接从 GT 提取）
- 没有场景感知（scene-aware）能力
- 没有处理噪声关键帧的鲁棒性机制

---

## 3. 最新相关方法综述

### 3.1 直接相关：Scene-Aware Motion In-betweening

#### SceneMI (ICCV 2025) ⭐⭐⭐ 最直接相关
- **核心思路**: 将 HSI 建模重新定义为 scene-aware motion in-betweening
- **关键创新**:
  - **Dual Scene Descriptors**: 同时编码全局场景上下文和局部场景上下文
  - **噪声关键帧鲁棒性**: 利用 diffusion 模型固有的去噪特性处理带噪声的关键帧
  - 在 TRUMANS 数据集上验证，可泛化到 GIMO（真实 IMU+手机采集的噪声数据）
- **应用场景**: 关键帧引导的场景内角色动画、不完美 HSI 数据的质量增强、单目视频 HSI 重建
- **与你的工作关系**: 几乎完全匹配你的需求——从粗提取的 HSI 关键帧做场景感知 in-betweening

#### Dyn-HSI (arXiv 2026.01)
- **核心思路**: 在动态场景中生成 HSI 动作，三大模块：
  - Vision: 动态场景感知导航
  - Memory: 层次化经验记忆
  - Control: HSI 扩散模型
- **关键创新**: 条件自回归扩散框架，支持动态变化的场景

### 3.2 Motion In-betweening 方法改进

#### sMDM - Sparse Motion Diffusion Model (ICCV 2025) ⭐⭐
- **核心思路**: 围绕稀疏、语义有意义的关键帧设计扩散模型
- **关键技术**:
  - 在 self-attention 中 mask 非关键帧，减少计算
  - 用 Lipschitz MLP 做平滑插值
  - **动态关键帧 mask 精炼**: 在推理过程中动态更新关键帧选择，早期用均匀 mask，后期选择最信息丰富的帧
- **启发**: 可以在去噪后期动态调整对粗关键帧的信任程度

#### SILK (2025)
- 证明简单的单 Transformer encoder 加上正确的数据处理（数据量、姿态表示、速度特征）即可达到竞争性结果
- **启发**: 数据工程（pose representation, velocity features）对质量影响巨大

#### AnyMoLe (CVPR 2025)
- 利用 video diffusion model 做任意角色的 motion in-betweening
- 两阶段帧生成 + ICAdapt 微调 + motion-video mimicking 优化

#### DNO - Diffusion Noise Optimization
- 在预训练 diffusion model 上优化 latent noise，无需重新训练即可做 motion denoising 和 completion
- **启发**: 可以用 test-time optimization 的方式精炼从粗关键帧生成的动作

### 3.3 物理合理性保证

#### PhysMoDPO (2026.03)
- 将 Whole-Body Controller 集成到 diffusion 训练中
- 使用 Direct Preference Optimization 对齐物理合理性和文本指令
- 不依赖手工 heuristic（如 foot sliding penalty）

#### POMP (CVPR 2025)
- 使用 phase manifold 对齐 motion prior 和物理约束
- 基于仿真的动态模块处理接触力、防止脚滑

#### FlexMotion (2025)
- Latent-space diffusion 集成关节位置、接触力、关节驱动和肌肉激活
- 轻量物理感知，不需要物理仿真器

#### SceMoS (2025)
- 用 2D 场景表示（鸟瞰图 + 局部高度图）做场景感知 3D 人体动作生成
- 细粒度接触推理 + 碰撞避免

### 3.4 HSI 重建与物理优化

#### HSImul3R (ICLR 2026) ⭐⭐
- **Physics-in-the-Loop 重建**: 将物理仿真器作为主动监督器
- **双向优化**:
  - 正向: 场景目标 RL 优化人体动作（运动保真度 + 接触稳定性）
  - 反向: Direct Simulation Reward Optimization 精炼场景几何
- **启发**: 物理仿真作为后处理精炼生成动作

---

## 4. 具体方案建议

### 方案 A: 基于 CondMDI 的渐进式改进（推荐起步方案）

在现有代码库上进行以下改进：

#### A1. 噪声关键帧的软约束处理
```
当前: 硬 imputation（直接替换）
改进: 加权 imputation + 置信度调度
```
- 为粗提取的关键帧引入置信度权重 `w_i ∈ [0,1]`
- 修改 imputation 策略：`x_t = w * x_t^{GT} + (1-w) * x_t^{model}`
- 置信度可基于: 提取方法的可信度、场景穿透程度、关节位置的不确定性

**具体修改点**: `diffusion/gaussian_diffusion.py` 中 imputation 逻辑

#### A2. Reconstruction Guidance 的改进
- 当前的 `grad_fn` 使用 L2 loss，对噪声关键帧敏感
- 改用 **Huber loss** 或 **adaptive weighting**，降低对离群关键帧的敏感度
- 引入 **gradient schedule**: 早期步骤用强约束保证全局结构，后期步骤放松约束让模型自由细化

**具体修改点**: `utils/editing_util.py` 中的 `grad_fn` 和 `get_gradient_schedule`

#### A3. 场景条件注入
- 在 MDM 的 Transformer 中增加场景编码的 cross-attention
- 场景编码可以是: 点云特征、SDF、BPS（Basis Point Set）或 occupancy features
- 参考 SceneMI 的 dual scene descriptor 设计

**具体修改点**: `model/mdm.py` 中的 `MDM` 类

### 方案 B: SceneMI 风格的 Scene-Aware In-betweening

完全参考 SceneMI 的架构设计：

1. **Dual Scene Descriptor**:
   - Global: 场景级别的 BEV 或点云特征
   - Local: 关键帧附近的局部场景特征（SDF 或接触概率图）

2. **噪声关键帧作为条件**:
   - 不将关键帧视为精确约束
   - 而是作为去噪的初始化或弱引导
   - diffusion 的去噪过程自然会修正噪声

3. **训练策略**:
   - 在训练时对 GT 关键帧人为添加噪声（模拟粗提取的误差）
   - 增强模型对不精确输入的鲁棒性

### 方案 C: Two-Stage: 生成 + 物理精炼

1. **Stage 1 - Diffusion 生成**: 使用方案 A/B 生成初始动作序列
2. **Stage 2 - 物理精炼**: 
   - 参考 HSImul3R 的 physics-in-the-loop 思路
   - 使用 RL-based motion tracking 在物理仿真器中追踪生成的动作
   - 自动解决穿透、浮空、脚滑等问题
   - 输出物理合理的最终动作

### 方案 D: Test-Time Optimization（DNO 风格）

1. 使用预训练的 CondMDI 模型
2. 在推理时优化 initial noise `z_T`:
   - 目标函数 = 关键帧重建 loss + 场景约束 loss（无穿透 + 接触合理）+ 动作平滑度
   - 梯度回传通过整个采样链
3. 优势: 不需要重新训练模型，灵活添加任意约束

---

## 5. 提升生成质量的关键技术点

### 5.1 数据层面
| 技术 | 说明 | 优先级 |
|------|------|--------|
| 训练时关键帧加噪 | 模拟粗提取误差，提升鲁棒性 | ⭐⭐⭐ |
| 速度特征 | 参考 SILK，velocity 是重要特征 | ⭐⭐⭐ |
| 数据增强 | 对 HSI 数据做几何增强（旋转、平移、镜像） | ⭐⭐ |
| 场景多样性 | 覆盖更多场景类型 | ⭐⭐ |

### 5.2 模型层面
| 技术 | 说明 | 优先级 |
|------|------|--------|
| Scene conditioning | 场景特征注入，避免穿透 | ⭐⭐⭐ |
| 置信度加权 | 对不同关键帧/关节用不同权重 | ⭐⭐⭐ |
| Classifier-Free Guidance | 场景/关键帧的 CFG 可提升条件遵循度 | ⭐⭐ |
| DiT 架构 | 代码库已有 `mdm_dit.py`，DiT 通常优于 vanilla Transformer | ⭐⭐ |

### 5.3 推理层面
| 技术 | 说明 | 优先级 |
|------|------|--------|
| Reconstruction Guidance 调度 | 动态调整引导强度 | ⭐⭐⭐ |
| 多次采样取最优 | Best-of-N sampling + 评分函数 | ⭐⭐ |
| Test-time optimization | 优化 noise 或 intermediate states | ⭐⭐ |
| 物理后处理 | 仿真器精炼 | ⭐ |

### 5.4 偏移量评估指标

#### 符号定义

设动作序列共 $T$ 帧，人体骨架共 $K$ 个关节。

| 符号 | 定义 |
|------|------|
| $\mathbf{J}\_{t,k}^{\text{gt}} \in \mathbb{R}^3$ | 第 $t$ 帧第 $k$ 个关节的 GT 三维位置 |
| $\mathbf{J}\_{t,k}^{\text{pred}} \in \mathbb{R}^3$ | 第 $t$ 帧第 $k$ 个关节的预测三维位置 |
| $\mathbf{x}\_t^{\text{gt}} = \mathbf{J}\_{t,0}^{\text{gt}} \in \mathbb{R}^3$ | 第 $t$ 帧 GT 根节点（pelvis）世界坐标 |
| $\mathbf{x}\_t^{\text{pred}} = \mathbf{J}\_{t,0}^{\text{pred}} \in \mathbb{R}^3$ | 第 $t$ 帧预测根节点世界坐标 |
| $\hat{\mathbf{J}}\_{t,k} = \mathbf{J}\_{t,k} - \mathbf{J}\_{t,0}$ | 根节点对齐后的局部关节位置 |
| $T$ | 序列总帧数 |
| $K$ | 关节总数 |

---

#### (1) 整体偏移量（Global Positional Offset, GPO）

衡量所有帧、所有关节在世界坐标系下的累积位置误差：

$$
\text{GPO} = \sum_{t=1}^{T} \left( \frac{1}{K} \sum_{k=1}^{K} \left\lVert \mathbf{J}\_{t,k}^{\text{gt}} - \mathbf{J}\_{t,k}^{\text{pred}} \right\rVert\_2 \right)
$$

等价地，若不取关节均值而是直接对所有关节求和：

$$
\text{GPO}^{\prime} = \sum_{t=1}^{T} \sum_{k=1}^{K} \left\lVert \mathbf{J}\_{t,k}^{\text{gt}} - \mathbf{J}\_{t,k}^{\text{pred}} \right\rVert\_2
$$

对应的 **逐帧均值版本**（MPJPE，Mean Per Joint Position Error）：

$$
\text{MPJPE} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{K} \sum_{k=1}^{K} \left\lVert \mathbf{J}\_{t,k}^{\text{gt}} - \mathbf{J}\_{t,k}^{\text{pred}} \right\rVert\_2
$$

---

#### (2) 轨迹偏移量（Trajectory Offset, TrajO）

仅衡量根节点在世界坐标系下的累积轨迹漂移：

$$
\text{TrajO} = \sum_{t=1}^{T} \left\lVert \mathbf{x}\_t^{\text{gt}} - \mathbf{x}\_t^{\text{pred}} \right\rVert\_2
$$

对应的均值版本：

$$
\overline{\text{TrajO}} = \frac{1}{T} \sum_{t=1}^{T} \left\lVert \mathbf{x}\_t^{\text{gt}} - \mathbf{x}\_t^{\text{pred}} \right\rVert\_2
$$

---

#### (3) 姿态偏移量（Pose Offset, PoseO）

衡量在去除全局平移后、纯粹局部姿态的误差。对每帧先做根节点对齐（root-aligned），再计算关节偏差：

$$
\hat{\mathbf{J}}\_{t,k}^{\text{gt}} = \mathbf{J}\_{t,k}^{\text{gt}} - \mathbf{J}\_{t,0}^{\text{gt}}, \quad \hat{\mathbf{J}}\_{t,k}^{\text{pred}} = \mathbf{J}\_{t,k}^{\text{pred}} - \mathbf{J}\_{t,0}^{\text{pred}}
$$

$$
\text{PoseO} = \sum_{t=1}^{T} \frac{1}{K} \sum_{k=1}^{K} \left\lVert \hat{\mathbf{J}}\_{t,k}^{\text{gt}} - \hat{\mathbf{J}}\_{t,k}^{\text{pred}} \right\rVert\_2
$$

对应的均值版本（即 root-aligned MPJPE，常记为 **RA-MPJPE**）：

$$
\text{RA-MPJPE} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{K} \sum_{k=1}^{K} \left\lVert \hat{\mathbf{J}}\_{t,k}^{\text{gt}} - \hat{\mathbf{J}}\_{t,k}^{\text{pred}} \right\rVert\_2
$$

> **备注**: GPO $\approx$ TrajO + PoseO。整体偏移可分解为轨迹漂移和局部姿态误差两个正交成分，方便定位问题来源。

---

#### (4) 首帧尾帧整体偏移量（Boundary Frame Offset, BFO）

衡量序列首帧（$t=1$）和尾帧（$t=T$）处的全局关节位置偏差，反映条件约束的遵循程度：

$$
\text{BFO} = \frac{1}{2} \sum_{t \in \{1, T\}} \frac{1}{K} \sum_{k=1}^{K} \left\lVert \mathbf{J}\_{t,k}^{\text{gt}} - \mathbf{J}\_{t,k}^{\text{pred}} \right\rVert\_2
$$

也可分别报告首帧和尾帧：

$$
\text{BFO}\_{\text{first}} = \frac{1}{K} \sum_{k=1}^{K} \left\lVert \mathbf{J}\_{1,k}^{\text{gt}} - \mathbf{J}\_{1,k}^{\text{pred}} \right\rVert\_2
$$

$$
\text{BFO}\_{\text{last}} = \frac{1}{K} \sum_{k=1}^{K} \left\lVert \mathbf{J}\_{T,k}^{\text{gt}} - \mathbf{J}\_{T,k}^{\text{pred}} \right\rVert\_2
$$

---

#### 指标汇总表

| 指标 | 缩写 | 公式核心 | 衡量内容 |
|------|------|---------|---------|
| 整体偏移量 | GPO / MPJPE | $\frac{1}{T}\frac{1}{K}\sum_t\sum_k\lVert\mathbf{J}^{gt}-\mathbf{J}^{pred}\rVert_2$ | 全局关节位置总误差 |
| 轨迹偏移量 | TrajO | $\frac{1}{T}\sum_t\lVert\mathbf{x}^{gt}-\mathbf{x}^{pred}\rVert_2$ | 根节点轨迹漂移 |
| 姿态偏移量 | PoseO / RA-MPJPE | $\frac{1}{T}\frac{1}{K}\sum_t\sum_k\lVert\hat{\mathbf{J}}^{gt}-\hat{\mathbf{J}}^{pred}\rVert_2$ | 去除平移后的纯姿态误差 |
| 首帧尾帧偏移量 | BFO | $\frac{1}{K}\sum_k\lVert\mathbf{J}^{gt}-\mathbf{J}^{pred}\rVert_2\big\|_{t\in\{1,T\}}$ | 边界帧条件约束遵循度 |

---

#### 其他辅助指标

| 指标 | 说明 |
|------|------|
| FID | 生成动作分布与 GT 分布的差距 |
| Diversity | 生成动作的多样性 |
| Foot Skating | 脚部滑动量 |
| Scene Penetration | 与场景的穿透距离 |
| Contact Accuracy | 接触点的准确率 |
| Jerk/Smoothness | 动作的平滑度 |

---

## 6. 推荐实施路线

```
Phase 1: 基础验证
├── 在 CondMDI 上直接用粗提取关键帧做 in-betweening（baseline）
├── 添加关键帧噪声增强的训练策略
└── 实现软约束 imputation

Phase 2: 场景感知
├── 设计场景编码器（BPS/SDF/PointNet++）
├── 在 MDM 中加入场景 cross-attention
└── 添加场景约束 loss（anti-penetration + contact）

Phase 3: 质量提升
├── 实现 reconstruction guidance 改进（Huber loss + adaptive schedule）
├── Test-time optimization (DNO 风格)
└── 物理精炼（可选）

Phase 4: 评估与迭代
├── 在 TRUMANS / GIMO 等 HSI 数据集上评估
├── 与 SceneMI 等 baseline 对比
└── 消融实验
```

---

## 7. 关键参考文献

1. **CondMDI** - Cohan et al., "Flexible Motion In-betweening with Diffusion Models", SIGGRAPH 2024 [本仓库]
2. **SceneMI** - Hwang et al., "Motion In-betweening for Modeling Human-Scene Interactions", ICCV 2025
3. **sMDM** - Bae et al., "Less is More: Improving Motion Diffusion Models with Sparse Keyframes", ICCV 2025
4. **Dyn-HSI** - "Dynamic Worlds, Dynamic Humans: Generating Virtual Human-Scene Interaction Motion in Dynamic Scenes", arXiv 2026
5. **HSImul3R** - "Physics-in-the-Loop Reconstruction of Simulation-Ready Human-Scene Interactions", ICLR 2026
6. **PhysMoDPO** - "Physically-Plausible Humanoid Motion with Preference Optimization", 2026
7. **POMP** - "Physics-consistent Motion Generative Model through Phase Manifolds", CVPR 2025
8. **SILK** - "Smooth InterpoLation frameworK for motion in-betweening", 2025
9. **AnyMoLe** - "Any Character Motion In-betweening Leveraging Video Diffusion Models", CVPR 2025
10. **DNO** - "Optimizing Diffusion Noise Can Serve As Universal Motion Priors", 2024
11. **TRUMANS** - Jiang et al., "Scaling Up Dynamic Human-Scene Interaction Modeling", CVPR 2024
12. **SceMoS** - "Scene-aware 3D Human Motion Generation using 2D Scene Representations", 2025
13. **FlexMotion** - "Lightweight Physics-Aware Motion Generation", 2025
