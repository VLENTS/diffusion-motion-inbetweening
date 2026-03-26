### 动作序列先验注入目标动作序列时产生显著偏移

#### 符号定义

- $T$：序列总帧数
- $K$：关节总数
- $\mathbf{J}_{t,k}^{gt}, \mathbf{J}_{t,k}^{pred} \in \mathbb{R}^3$：第 $t$ 帧第 $k$ 个关节的 GT / 预测三维位置
- $\mathbf{x}_t = \mathbf{J}_{t,0} \in \mathbb{R}^3$：第 $t$ 帧根节点（pelvis）坐标
- $\hat{\mathbf{J}}_{t,k} = \mathbf{J}_{t,k} - \mathbf{J}_{t,0}$：根节点对齐后的局部关节位置

#### 评估指标

**整体偏移量（MPJPE）**：所有帧所有关节在世界坐标系下的平均位置误差

$$\text{MPJPE} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{K} \sum_{k=1}^{K} \left\lVert \mathbf{J}_{t,k}^{gt} - \mathbf{J}_{t,k}^{pred} \right\rVert_2$$

**轨迹偏移量（TrajE）**：根节点在世界坐标系下的平均轨迹漂移

$$\text{TrajE} = \frac{1}{T} \sum_{t=1}^{T} \left\lVert \mathbf{x}_t^{gt} - \mathbf{x}_t^{pred} \right\rVert_2$$

**姿态偏移量（RA-MPJPE）**：每帧先对齐根节点消除全局平移，再计算局部姿态误差

$$\hat{\mathbf{J}}_{t,k}^{gt} = \mathbf{J}_{t,k}^{gt} - \mathbf{J}_{t,0}^{gt}, \quad \hat{\mathbf{J}}_{t,k}^{pred} = \mathbf{J}_{t,k}^{pred} - \mathbf{J}_{t,0}^{pred}$$

$$\text{RA-MPJPE} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{K} \sum_{k=1}^{K} \left\lVert \hat{\mathbf{J}}_{t,k}^{gt} - \hat{\mathbf{J}}_{t,k}^{pred} \right\rVert_2$$

**首帧尾帧整体偏移量（BFE）**：边界帧处全关节平均位置误差

$$\text{BFE} = \frac{1}{2}\left( \frac{1}{K}\sum_{k=1}^{K}\left\lVert \mathbf{J}_{1,k}^{gt} - \mathbf{J}_{1,k}^{pred} \right\rVert_2 + \frac{1}{K}\sum_{k=1}^{K}\left\lVert \mathbf{J}_{T,k}^{gt} - \mathbf{J}_{T,k}^{pred} \right\rVert_2 \right)$$

> **诊断关系**：MPJPE 可分解为 TrajE（轨迹漂移）+ RA-MPJPE（局部姿态误差），用于定位偏移来源。BFE 单独衡量关键帧约束的遵循程度。

#### 原因分析

##### 目标动作序列在先验分布外

**验证方法**: 使用三种互补手段判定目标序列是否在模型学习的先验分布之外。

**方法 1: Embedding 空间可视化 (t-SNE / UMAP)**

将训练集和目标序列通过 MDM Transformer encoder 提取全局 embedding，降维到 2D 可视化：
- 若目标序列（红色）聚集在训练集（蓝色）覆盖区域内 → 分布内
- 若目标序列散布在训练集覆盖区域之外 → OOD

**方法 2: 逐样本 Diffusion 重建损失**

对每个样本在多个 diffusion timestep 上计算模型重建损失 $\mathcal{L}(x_0)$：

$$\mathcal{L}(x_0) = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \left\lVert x_0 - \hat{x}_0(x_t, t) \right\rVert^2$$

其中 $\mathcal{T}$ 为均匀采样的时间步子集，$\hat{x}_0$ 为模型预测。OOD 样本的重建损失通常显著高于训练分布内样本。

**方法 3: 特征空间统计距离**

在 embedding 空间计算两组定量指标：
- **马氏距离**: 每个目标样本到训练集分布（$\mu_{\text{train}}, \Sigma_{\text{train}}$）的马氏距离
- **k-NN 距离**: 每个目标样本到训练集最近 $k$ 个邻居的平均欧氏距离

**OOD 判定参考标准**:
| 指标比值（目标/训练） | 分布内 | 边界 | 分布外 |
|---|---|---|---|
| 重建损失比 | < 1.2x | 1.2x ~ 1.5x | > 1.5x |
| 马氏距离比 | < 1.5x | 1.5x ~ 2.0x | > 2.0x |
| k-NN 距离比 | < 1.5x | 1.5x ~ 2.0x | > 2.0x |

**关于文本条件的处理**:

脚本默认开启 `--uncond_text`，通过 `mask_cond(force_mask=True)` 将文本 embedding 置零。这与模型 forward 中 `y['uncond']=True` 的 CFG 无条件分支完全一致，确保只比较运动先验分布，排除文本语义的干扰。

> 注意: 传空串 `''` 仍然会经 CLIP 编码得到非零向量，**不等于**数学意义上的无条件。

**运行脚本**:

```bash
# 默认: 关闭文本条件（推荐，只比较运动先验）
python scripts/verify_ood_distribution.py \
    --model_path save/condmdi/model000500000.pt \
    --target_motion_path /path/to/target_motions.npy \
    --output_dir save/ood_analysis

# 可选: 保留 CLIP 空串编码
python scripts/verify_ood_distribution.py \
    --model_path save/condmdi/model000500000.pt \
    --target_motion_path /path/to/target_motions.npy \
    --output_dir save/ood_analysis \
    --no_uncond_text
```

输出文件:
- `distribution_tsne.png` — t-SNE 分布可视化
- `loss_distribution.png` — 重建损失直方图
- `distance_comparison.png` — 马氏距离 & k-NN 距离对比图
- `ood_report.txt` — 综合诊断报告（标注了文本条件模式）
- `raw_metrics.npz` — 原始数值

**实验结果**:

验证设定:
- 训练参考 = 全量 train.txt 去掉 exclude_range 中的 id
- 目标 = target_range 对应序列
- exclude_range: 300001-300013, target_range: 300005-300013

```
训练集样本数: 500
目标序列数:   9
文本条件:     关闭 (uncond)

[重建损失]
  训练集均值: 0.133541 ± 0.128599
  目标均值:   0.214753 ± 0.102504
  比值:       1.61x

[马氏距离]
  训练集到自身: 14.8838 ± 4.4436
  目标到训练集: 27.1814 ± 7.8151
  比值:         1.83x

[k-NN 距离]
  训练集内部: 4.0257 ± 2.5118
  目标到训练: 4.5049 ± 0.9057
  比值:       1.12x
```

**结果解读**:

三个指标呈现出不一致的信号，这本身揭示了 OOD 的具体模式：

| 指标 | 比值 | 判定 | 含义 |
|------|------|------|------|
| 重建损失 | 1.61x | OOD | 模型对这些序列的去噪预测能力显著弱于训练集 |
| 马氏距离 | 1.83x | 边界 OOD | 在全局分布形状（均值+协方差）上偏离训练集 |
| k-NN 距离 | 1.12x | 分布内 | 在局部邻域上与训练集某些样本接近 |

这组指标组合指向一个典型模式：**目标序列处于训练分布的低密度尾部区域**。

- k-NN 距离接近（1.12x）说明目标序列并不是一种模型"完全没见过"的运动类型，它们在局部邻域能找到相似的训练样本
- 但马氏距离偏高（1.83x）说明它们偏离了训练分布的高密度中心区域
- 重建损失最高（1.61x）说明模型对这个区域的建模精度不足——训练过程中这类样本出现频率低，模型没有充分学习

> 类比：不是"没见过猫"（全新类别），而是"只见过家猫，现在来了一只姿态罕见的猫"（同分布低密度尾部）。

**关于 t-SNE 看不出差异的原因**:

这是预期之中的：
1. **样本量极度不平衡**：9 个目标点 vs 500 个训练点，在 2D 投影中极易被"淹没"
2. **t-SNE 不保距**：t-SNE 优化局部邻域结构，会扭曲全局密度差异；两个在高维空间距离较远的点在 2D 中可能被拉到一起
3. **高维到 2D 信息损失巨大**：512 维 embedding → 2 维，大量区分信息被丢弃

t-SNE/UMAP 在此场景下是**辅助参考**，定量指标（重建损失 + 马氏距离）更可靠。

**结论**: 轻度到中度 OOD。目标序列处于训练分布的低密度尾部，不是分布外的全新类别。三个指标中仅重建损失（1.61x）刚过 OOD 阈值，马氏距离（1.83x）在边界区，k-NN（1.12x）明确在分布内。"目标序列在先验分布外"可以作为偏移的部分解释，但 OOD 程度不严重，不是主要原因。更需要关注第二个假设——模型条件控制能力不足。

##### 提供先验的模型在条件控制能力上表现差

**实验数据**:

| 配置 | MPJPE | TrajE | RA-MPJPE | BFE |
|------|-------|-------|----------|-----|
| line_a: `benchmark_sparse` T=5, impute, stop=0 | 0.0505 | 0.0497 | 0.0092 | 0.0137 |
| line_d: manual kf, impute, stop=0 | 0.0394 | 0.0000 | 0.0394 | 0.0284 |

**诊断: Line A 的误差构成**

| 成分 | 值 | 占 MPJPE 比例 |
|------|----|-------------|
| TrajE（轨迹漂移） | 0.0497 | **98.4%** |
| RA-MPJPE（姿态误差） | 0.0092 | 1.6% |

Line A 的核心问题不是姿态——RA-MPJPE 已经接近 0.01 级别了。**几乎全部误差来自根节点轨迹漂移。**

`benchmark_sparse` + `transition_length=5` 意味着每 5 帧给一个全关节关键帧。在关键帧上模型会被 imputation 强制对齐到 GT，但在两个关键帧之间的 4 帧里，模型对根节点的预测产生了累积漂移。由于 HumanML 的根节点表示是相对位移（帧间增量），这种漂移在长序列上会被放大。

**与 Line D 的对比**:

| | Line A | Line D |
|---|---|---|
| 轨迹 | 漂移严重 (0.0497) | 完美 (0.0000) |
| 姿态 | 很好 (0.0092) | 较差 (0.0394) |
| 策略 | 稀疏全关节关键帧 | 手动关键帧 + 指定关节 |

Line D 用连续帧的根节点约束锁死了轨迹，但关节覆盖不全导致姿态误差大。
Line A 用稀疏全关节关键帧保住了姿态，但关键帧间距太大导致轨迹漂移。

**两条线各自拿到了 <0.01 的一半**：Line A 的姿态、Line D 的轨迹。

---

**降低 MPJPE 至 < 0.01 的策略**

**策略 1: 全帧根轨迹约束 + 稀疏全关节关键帧（已实现，首选）**

同时施加两层 imputation mask：
- 第一层：**所有帧** 的 **根节点（pelvis）** 特征 → 锁死轨迹（类似 Line D 的 TrajE=0）
- 第二层：**每 N 帧** 的 **全关节** 特征 → 保持姿态（类似 Line A 的 RA-MPJPE=0.009）

已在 `utils/editing_util.py` 中新增 `edit_mode='pelvis_dense_sparse'`。

使用方式：

```bash
--edit_mode pelvis_dense_sparse --transition_length 5 \
--imputate --stop_imputation_at 0
```

预期效果：TrajE ≈ 0 + RA-MPJPE ≈ 0.009 → MPJPE ≈ 0.009 < 0.01

**策略 2: 加密关键帧间距（简单但暴力）**

将 `transition_length` 从 5 降到 2 或 1：
- `transition_length=2`：每 2 帧一个关键帧，非关键帧只有 1 帧间距，轨迹漂移极小
- `transition_length=1`：每帧都是关键帧，imputation 将直接复制 GT（MPJPE → 0，但失去生成意义）

权衡：间距越小约束越强，但生成自由度越低。`transition_length=2~3` 可能是平衡点。

**策略 3: Imputation + Reconstruction Guidance 双管齐下**

在 Line A 基础上叠加 reconstruction guidance，专门针对根节点轨迹加强引导：

```bash
--imputate --stop_imputation_at=0 \
--reconstruction_guidance --reconstruction_weight=10 \
--gradient_schedule=exponential
```

reconstruction guidance 通过梯度将非关键帧的根轨迹也拉向 GT 方向，弥补 imputation 的间隙。

**实验进展（Line A → D → G）**:

| 配置 | MPJPE | TrajE | RA-MPJPE | BFE | 关键帧策略 |
|------|-------|-------|----------|-----|-----------|
| line_a | 0.0505 | 0.0497 | 0.0092 | 0.0137 | 全关节 stride=5 |
| line_d | 0.0394 | 0.0000 | 0.0394 | 0.0284 | root stride=1 + 9关节 stride=40 |
| line_g | 0.0163 | 0.0000 | 0.0163 | 0.0077 | root stride=1 + 9关节 stride=5 |

趋势分析：
- D→G：joint stride 40→5，RA-MPJPE 0.039→0.016（关键帧密度有效）
- G vs A：同为 stride=5，但 G 只有 9/22 关节，A 有 22/22 关节，RA-MPJPE 差 0.007
- **残余误差完全来自未观测的 13 个关节**（脊柱链、膝、肘、肩、颈、头）

**降至 MPJPE < 0.01 的调整方案**:

在 Line G 基础上，保持 root stride=1 不变：

| 调整 | 做法 | 预期 MPJPE |
|------|------|-----------|
| 扩大关节覆盖至全部 22 | 去掉 `--manual_observed_joints` 限制，joint stride=5 | ~0.009 |
| 全关节 + stride=3 | 全关节，`--manual_observed_joint_stride 3` | ~0.005-0.007 |

**策略优先级**:

| 策略 | 预期 MPJPE | 实现难度 | 推荐 |
|------|-----------|---------|------|
| root stride=1 + 全关节 stride=5 | ~0.009 | 改参数 | 首选 |
| root stride=1 + 全关节 stride=3 | ~0.005-0.007 | 改参数 | 更激进 |
| 上述 + reconstruction guidance | 再降 ~20-30% | 加参数 | 叠加改善 |
