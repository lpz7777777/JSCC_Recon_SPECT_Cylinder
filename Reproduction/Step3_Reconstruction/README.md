# 步骤3：图像重建

## 概述

本步骤使用步骤2生成的CntStat（正弦图）或List（康普顿事件列表），通过OSEM（有序子集期望最大化）迭代算法进行图像重建。

提供两种重建模式：
- **SC重建**（Single Photon Counting）：仅使用单光子计数数据
- **JSCCSD重建**（Joint Compton Scatter Decode）：利用康普顿散射信息的联合解码重建

## 前置条件

### 数据文件准备

在运行之前，需要将以下文件放入对应目录：

```
Step3_Reconstruction/
├── Factors/              ← 从步骤1复制（极坐标系统矩阵等）
│   └── <energy>keV_RotateNum<rotate_num>/
│       ├── SysMat_polar
│       ├── coor_polar_full.csv
│       ├── RotMat_full.csv
│       └── RotMatInv_full.csv
├── CntStat/              ← 从步骤2复制（正弦图数据）
│   └── CntStat_<phantom>_<count>.csv
└── List/                 ← 从步骤2复制（康普顿事件列表）
    └── List_<phantom>_<count>/
        └── <rotate_id>.csv
```

### Python 环境

```bash
pip install torch numpy
```

## 模式A：SC重建（仅单光子计数）

### 代码文件
```
SC_Recon/
├── main_local_cntstat.py          # 主入口脚本
├── recon_osem_local_cntstat.py    # OSEM重建核心算法
├── compton_sparse_ops.py          # 康普顿稀疏操作工具
├── sparse_main_utils.py           # 稀疏重建通用工具
├── process_list_plane_sparse.py   # 事件列表处理（稀疏模式）
└── process_list_plane_strict.py   # 事件列表处理（严格模式）
```

### 运行方法
```bash
cd SC_Recon
python main_local_cntstat.py \
    --e0-list 0.511 \
    --data-file-name ContrastPhantom_240_30 \
    --count-level 1e9 \
    --sc-iter 10000 \
    --save-iter-step 100
```

### 关键参数说明
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--e0-list` | 光子能量（MeV） | 0.511（511keV）/ 0.140（140keV） |
| `--data-file-name` | 数据文件名前缀 | ContrastPhantom_240_30 |
| `--count-level` | 总计数水平 | 1e9 |
| `--sc-iter` | SC重建迭代次数 | 10000 |
| `--save-iter-step` | 每隔多少次迭代保存一次 | 100 |

### 输出
重建结果保存在 `Figure_Dist_SC/` 目录，包含：
- `Image_SC_Iter_<iterMax>_<saveCount>` — 各迭代的重建图像（float32二进制）
- `Cartesian/` — 极坐标到直角坐标转换后的图像

## 模式B：JSCCSD重建（联合康普顿散射解码）

### 代码文件
```
JSCCSD_Recon/
├── main_local_sparse_jsccsd_only.py   # 主入口脚本
├── recon_osem_local_sparse_jsccsd_only.py  # JSCCSD重建核心算法
├── compton_sparse_ops.py              # 康普顿稀疏操作工具
├── sparse_main_utils.py               # 稀疏重建通用工具
├── process_list_plane_sparse.py       # 事件列表处理（稀疏模式）
└── process_list_plane_strict.py       # 事件列表处理（严格模式）
```

### 运行方法
```bash
cd JSCCSD_Recon
python main_local_sparse_jsccsd_only.py \
    --e0-list 0.511 \
    --data-file-name ContrastPhantom_240_30 \
    --count-level 1e9 \
    --jsccsd-iter 5000 \
    --save-iter-step 50
```

### 关键参数说明
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--e0-list` | 光子能量（MeV） | 0.511 |
| `--data-file-name` | 数据文件名前缀 | ContrastPhantom_240_30 |
| `--count-level` | 总计数水平 | 1e9 |
| `--jsccsd-iter` | JSCCSD重建迭代次数 | 5000 |
| `--save-iter-step` | 每隔多少次迭代保存一次 | 50 |

### 输出
重建结果保存在 `Fig_JSCC/` 目录，包含：
- `Image_JSCCSD_Iter_<iterMax>_<saveCount>` — 各迭代的重建图像
- `Cartesian/` — 极坐标到直角坐标转换后的图像

## 依赖文件说明

### compton_sparse_ops.py
康普顿散射相关的稀疏矩阵操作，包括：
- 康普顿散射系统矩阵的构建
- 稀疏矩阵的GPU加速计算

### sparse_main_utils.py
稀疏重建的通用工具函数，包括：
- 数据加载和预处理
- 极坐标/直角坐标转换
- 迭代结果的保存和管理

### process_list_plane_sparse.py / process_list_plane_strict.py
事件列表处理模块，用于将原始康普顿事件列表转换为重建可用的格式。

## 注意事项

1. **GPU推荐**：重建过程涉及大量矩阵运算，强烈建议使用CUDA GPU
2. **内存需求**：系统矩阵可能较大，建议至少16GB内存
3. **路径注意**：脚本中使用相对路径引用 `../Factors/`、`../CntStat/`、`../List/` 等目录
4. **迭代次数**：SC重建通常需要更多迭代（~10000），JSCCSD收敛更快（~5000）

## 下一步

将重建结果目录（如 `Figure_Dist_SC/Cartesian/`）复制到 `Step4_Visualization/Figure_Dist_SC/` 或 `Step4_Visualization/Figure_Dist_JSCCSD/`，然后继续步骤4。