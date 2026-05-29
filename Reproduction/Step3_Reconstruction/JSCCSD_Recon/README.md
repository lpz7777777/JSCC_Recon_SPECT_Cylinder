# JSCCSD 重建（Joint Compton Scatter Decode Reconstruction）

## 概述

本目录实现了联合康普顿散射解码（JSCCSD）重建算法。与SC重建不同，JSCCSD同时利用光电峰计数和康普顿散射事件信息，通过稀疏事件列表和粗-细网格投影实现高效计算。

## 文件列表

| 文件 | 功能 |
|------|------|
| `main_local_sparse_jsccsd_only.py` | 主入口脚本 |
| `recon_osem_local_sparse_jsccsd_only.py` | JSCCSD重建核心算法 |
| `compton_sparse_ops.py` | 康普顿稀疏操作：粗细网格投影、事件物化 |
| `sparse_main_utils.py` | 通用工具函数 |
| `process_list_plane_sparse.py` | 事件列表处理（稀疏模式） |
| `process_list_plane_strict.py` | 事件列表处理（严格模式） |

## 算法原理

### 1. 联合重建总体框架

JSCCSD同时利用两种测量数据：

| 数据类型 | 说明 | 权重系数 |
|----------|------|----------|
| **光电峰计数（SC）** | 能量窗口内的单光子计数 | `α` |
| **康普顿散射事件（List）** | 每个事件的散射路径概率分布 | `2 - α` |

`α ∈ [1, 2]` 是平衡系数：
- `α = 2`：退化为纯SC重建
- `α = 1`：SC和康普顿等权重
- `α = 1.5`（推荐）：平衡两种信息

每次迭代：`weight_total = α · weight_SC + (2 - α) · weight_compton`

### 2. 康普顿反投影建模

#### 2.1 问题背景

光子可能在探测器内先发生康普顿散射再被光电吸收。对每个散射事件，已知：
- **cpnum1**：第一次相互作用的探测器bin编号
- **t_compton**：该事件在各粗网格像素上的概率分布

#### 2.2 粗-细网格系统（compton_sparse_ops.py）

极坐标网格在不同半径上角度采样数不同（内环40，外环160），直接在全分辨率上计算代价极大。

解决方案：构建粗网格存储事件概率，运行时上采样到细网格。

```
细网格（fine）              粗网格（coarse）
角度：40~160/环    →→→     角度：20~80/环（步长theta_stride=2）
Z轴：20层          →→→     Z轴：10层（步长z_stride=2）
```

**ComptonSparseProjector** 数据结构包含：
- `layer_reduce / layer_expand`：角度方向的降采样/上采样矩阵（周期性）
- `z_reduce / z_expand`：Z方向的降采样/上采样矩阵（线性插值）

#### 2.3 事件物化（materialize_sparse_event_rows_to_fine）

康普顿反投影的核心函数：

```python
def materialize_sparse_event_rows_to_fine(event_block, sysmat, projector):
    cpnum1, t_compton = unpack(event_block)
    t_compton_fine = upsample_coarse_to_fine(t_compton, projector)  # 粗→细上采样
    sysmat_rows = sysmat[cpnum1, :]                                 # 提取bin系统矩阵行
    t_fine = t_compton_fine * sysmat_rows                           # 联合概率
    t_fine = t_fine / row_sum(t_fine)                               # 归一化
    return t_fine
```

物理解释：
- `sysmat[cpnum1]`：光子从各体素到达第一次作用bin的概率
- `t_compton_fine`：散射发生在各体素位置的概率
- 相乘：光子到达体素j并散射回bin的联合概率
- 归一化后：给定散射事件，光子来自各体素的后验概率

#### 2.4 康普顿权重计算

```python
def get_weight_compton_sparse(event_block, sysmat_full, img_rotate, projector):
    t_fine = materialize_sparse_event_rows_to_fine(...)  # 物化事件
    denom = t_fine @ img_rotate                           # 正向投影
    weight = t_fine^T @ (1 / denom)                       # 反投影
    return weight
```

与SC的区别：每个事件贡献权重 `1/denom`（非计数值 `y_i/forward_i`），因为每个事件是独立的。

### 3. 联合OSEM迭代流程

```python
for subset in subsets:
    weight_total = 0

    # SC部分：α × 标准反投影校正
    for rotate_idx in rotate_num:
        w_s = alpha * (sysmat^T @ (proj / (sysmat @ img_rotate)))
        weight_total += w_s（逆旋转映射）

    # 康普顿部分：(2-α) × 稀疏事件反投影
    for rotate_idx in rotate_num:
        for t_block in event_blocks[rotate_idx]:
            w_c = (2-alpha) * get_weight_compton_sparse(t_block, ...)
            weight_total += w_c（逆旋转映射）

    # EM更新
    img = safe_em_update(img, weight_total, s_map)
```

### 4. 事件列表分块处理

```
t_list_all[subset][rotate][divide][energy] = event_block
```

三级分块控制GPU内存：subset（OSEM子集）→ rotate（角度）→ divide（事件块）

### 5. 安全EM更新

处理数值不稳定（NaN/Inf/除零）：
```python
valid = s_map > eps
img[valid] = img[valid] * clamp(weight[valid], min=0) / s_map[valid]
```

## SC vs JSCCSD 对比

| 特性 | SC重建 | JSCCSD重建 |
|------|--------|------------|
| 使用数据 | CntStat（计数） | CntStat + List |
| 康普顿散射 | 丢弃 | 利用 |
| 计算量 | 较小 | 较大（事件物化） |
| 收敛速度 | ~10000迭代 | ~5000迭代 |
| 统计噪声 | 较高 | 较低 |
| CRC/CNR | 基准 | 显著提升 |

## 运行方法

```bash
python main_local_sparse_jsccsd_only.py \
    --e0-list 0.511 \
    --data-file-name ContrastPhantom_240_30 \
    --count-level 1e9 \
    --jsccsd-iter 5000 \
    --save-iter-step 50
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--jsccsd-iter` | 5000 | 总迭代次数 |
| `--alpha` | 1.5 | SC与康普顿的权重平衡 |
| `--theta-stride` | 2 | 角度粗网格步长 |
| `--z-stride` | 2 | Z方向粗网格步长 |
| `--t-divide-num` | - | 事件分块数（控制GPU内存） |