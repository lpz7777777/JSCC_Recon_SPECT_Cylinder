# SC 重建（Single Photon Counting Reconstruction）

## 概述

本目录实现了基于OSEM算法的极坐标SPECT单光子计数重建。仅使用光电峰能量窗口内的计数数据，不利用康普顿散射信息。

## 文件列表

| 文件 | 功能 |
|------|------|
| `main_local_cntstat.py` | 主入口脚本：参数解析、数据加载、调用重建 |
| `recon_osem_local_cntstat.py` | OSEM重建核心算法实现 |
| `compton_sparse_ops.py` | 康普顿稀疏操作工具（SC模式不使用康普顿部分） |
| `sparse_main_utils.py` | 通用工具函数：数据加载、坐标转换、结果保存 |
| `process_list_plane_sparse.py` | 事件列表处理（稀疏模式） |
| `process_list_plane_strict.py` | 事件列表处理（严格模式） |

## 算法原理

### 1. OSEM 重建算法

OSEM（Ordered Subset Expectation Maximization）是ML-EM的加速版本，将投影数据分成若干子集，每个子集进行一次图像更新。

#### ML-EM 基本公式

测量模型：`y_i = Σ_j H_ij · x_j`

- `y_i`：探测器bin `i` 的测量计数
- `x_j`：图像体素 `j` 的活度值
- `H_ij`：系统矩阵元素

ML-EM 迭代更新：

```
x_j^(k+1) = x_j^(k) · [ Σ_i H_ij · y_i / ŷ_i^(k) ] / [ Σ_i H_ij ]
```

- `ŷ_i^(k) = Σ_j H_ij · x_j^(k)` 正向投影（预测计数）
- 分子：反投影的比值校正
- 分母：灵敏度图（sensitivity map）

#### OSEM 子集策略

将探测器bin随机分成 `N_subset` 个子集，每次用一个子集更新：

```python
for subset in random_subsets:
    forward = sysmat @ img_rotate              # 正向投影
    ratio = proj / forward                      # 实测/预测
    weight = sysmat^T @ ratio                   # 反投影
    img = img * weight / s_map                  # EM更新
```

### 2. 极坐标旋转映射

极坐标下旋转不是系统矩阵变化，而是像素索引映射：

- **RotMat `[pixel_num × rotate_num]`**：`RotMat[p, r]` = 像素p在旋转角度r时的索引
- **RotMatInv**：逆映射，将反投影结果映射回原始像素顺序

```python
for rotate_idx in range(rotate_num):
    img_rotate = img[RotMat[:, rotate_idx]]       # 旋转图像
    forward = sysmat @ img_rotate                  # 正向投影
    weight = sysmat^T @ (proj / forward)           # 反投影
    weight_local += weight[RotMatInv[:, rotate_idx]]  # 逆旋转
```

**优势**：系统矩阵只需存储一个角度，通过旋转映射处理所有角度，大幅减少内存。

### 3. 随机bin子集划分

`build_random_bin_subsets()` 随机打乱bin索引后均分为子集，每次迭代重新打乱（随机排序OSEM），避免相邻bin相关性影响收敛。

### 4. 灵敏度图

`s_map[j] = Σ_i Σ_r H_ij`，所有旋转角度和探测器bin对体素j的总灵敏度，用作归一化因子。

### 5. 多能量窗口

支持多能量窗口：`sysmat_all = [sysmat_e1, sysmat_e2, ...]`，各窗口校正因子累加到同一权重。

## 数据流

```
输入 → SysMat_polar, CntStat, RotMat_full, RotMatInv_full, s_map
迭代 → 初始化img=1 → 随机划分子集 → 正向投影 → 反投影 → EM更新
输出 → Image_SC, Image_SC_Iiter_<max>_<count>
```

## 运行方法

```bash
python main_local_cntstat.py \
    --e0-list 0.511 \
    --data-file-name ContrastPhantom_240_30 \
    --count-level 1e9 \
    --sc-iter 10000 \
    --save-iter-step 100