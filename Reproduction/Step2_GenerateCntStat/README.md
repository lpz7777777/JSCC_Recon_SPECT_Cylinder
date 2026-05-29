# 步骤2：生成CntStat（正弦图）和List（事件列表）

## 概述

本步骤使用步骤1生成的Factors文件，通过正向投影生成模拟测量数据，包括：
- **CntStat**：每个探测器bin在每个旋转角度下的单光子计数（正弦图）
- **List**：康普顿散射事件的详细列表（用于JSCCSD重建）

## 前置条件

1. 步骤1已完成，生成了以下Factors文件：
   - `SysMat_polar`（极坐标系统矩阵）
   - `coor_polar_full.csv`（极坐标坐标）
   - `RotMat_full.csv` / `RotMatInv_full.csv`（旋转矩阵）
2. Geant4 模拟生成的投影数据文件（如 Phantom_RawCompressed.mat）

## 运行方法

在 MATLAB 中运行：
```matlab
GenProj_SPECT_PolarCoor
```

## 脚本说明

### GenProj_SPECT_PolarCoor.m

核心正向投影脚本，主要功能：
1. 加载极坐标系统矩阵（SysMat_polar）
2. 加载旋转矩阵（RotMat_full, RotMatInv_full）
3. 定义模体（如对比度模体 ContrastPhantom）
4. 通过系统矩阵计算正向投影，生成各旋转角度的投影数据
5. 对投影数据添加泊松噪声（根据设定的计数水平）
6. 生成并保存CntStat和List文件

**关键参数说明**：
- `e0_list`：光子能量（如0.511表示511keV, 0.140表示140keV）
- `data_file_name`：模体名称
- `count_level`：总计数水平（如1e9表示10亿次计数）
- `rotate_num`：旋转角度数（通常为20）

## 输出文件

生成的CntStat文件（CSV格式），包含每个旋转角度下每个极坐标bin的计数统计。

生成的List文件包含康普顿散射事件的详细信息（用于步骤3的JSCCSD重建）。

## 下一步

将生成的文件复制到 `Step3_Reconstruction/` 对应目录：
- CntStat文件 → `Step3_Reconstruction/CntStat/`
- List文件 → `Step3_Reconstruction/List/`

然后继续步骤3。