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

##### 提供先验的模型在条件控制能力上表现差
