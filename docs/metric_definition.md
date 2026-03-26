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

**运行脚本**:

```bash
python scripts/verify_ood_distribution.py \
    --model_path save/condmdi/model000500000.pt \
    --target_motion_path /path/to/target_motions.npy \
    --output_dir save/ood_analysis
```

输出文件:
- `distribution_tsne.png` — t-SNE 分布可视化
- `loss_distribution.png` — 重建损失直方图
- `distance_comparison.png` — 马氏距离 & k-NN 距离对比图
- `ood_report.txt` — 综合诊断报告
- `raw_metrics.npz` — 原始数值

##### 提供先验的模型在条件控制能力上表现差
