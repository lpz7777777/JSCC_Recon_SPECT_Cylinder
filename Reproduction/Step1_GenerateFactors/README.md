# 步骤1：生成Factors文件

## 概述

本步骤从 Monte Carlo 模拟生成的原始系统矩阵（SysMat.sysmat）出发，筛选有效晶体（CrystalMatrix中=1的bin），并将直角坐标系统矩阵转换为极坐标系统矩阵，生成重建所需的全部Factors文件。

## 前置条件

需要在 `Data/` 目录中准备以下文件：

| 文件名 | 说明 |
|--------|------|
| `CrystalMatrix_20250307_JSCCGC_32x64x4.mat` | 晶体矩阵定义文件（32×64×4 探测器） |
| `SysMat.sysmat` | Monte Carlo 模拟生成的原始系统矩阵（二进制float32，形状[51,51,20,总bin数]） |

## 文件说明

### PrintCrystalMatrix.m（可选，验证用）
将 CrystalMatrix 从 .mat 格式导出为 .txt 文本格式，方便查看和验证晶体矩阵的内容。

### genSysMatPolar_3D.m（核心脚本，必须运行）
一体化脚本，完成以下全部工作：
1. 加载 CrystalMatrix 定义，筛选 CrystalMatrix==1 的 bin
2. 从 `Data/SysMat.sysmat` 读取原始系统矩阵，提取筛选后的直角坐标系统矩阵
3. 定义极坐标网格（径向 + 角度）
4. 通过双线性插值将系统矩阵从直角坐标转换到极坐标
5. 生成旋转映射矩阵（RotMat / RotMatInv）
6. 输出所有 Factors 文件

**极坐标参数**：
- 径向范围：3mm 到 150mm，步长 3mm（50个环）
- 角度分辨率：每环 40 到 160 个角度，随半径增大而增多
- 旋转数：20
- Z轴切片：20层

## 运行顺序

### 1.（可选）运行 PrintCrystalMatrix.m
验证晶体矩阵内容：
```matlab
PrintCrystalMatrix
```

### 2. 运行 genSysMatPolar_3D.m
生成所有Factors文件：
```matlab
genSysMatPolar_3D
```

运行时间取决于系统矩阵大小，通常需要几分钟到十几分钟。

## 输出文件

运行完成后，在当前目录下生成以下文件：

| 文件名 | 格式 | 说明 |
|--------|------|------|
| `SysMat_polar` | 二进制float32 | 极坐标系统矩阵，形状为 [num_bins, pixel_num_per_ring, num_z] |
| `coor_polar.csv` | CSV | 单层极坐标像素坐标 [x, y] |
| `coor_polar_full.csv` | CSV | 完整极坐标像素坐标 [x, y, z] |
| `RotMat.csv` | CSV | 旋转映射矩阵 |
| `RotMatInv.csv` | CSV | 旋转逆映射矩阵 |
| `RotMat_full.csv` | CSV | 全Z轴旋转映射矩阵 |
| `RotMatInv_full.csv` | CSV | 全Z轴旋转逆映射矩阵 |

## 下一步

将生成的以下文件复制到 `Step3_Reconstruction/Factors/<energy>keV_RotateNum<rotate_num>/` 目录：
- `coor_polar_full.csv`
- `RotMat_full.csv`
- `RotMatInv_full.csv`
- `SysMat_polar`

然后继续步骤2。