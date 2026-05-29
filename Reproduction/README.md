# JSCC-SPECT 圆柱面探测器极坐标重建 — 完整复现指南

## 项目简介

本项目实现了基于联合康普顿散射编码（JSCC）的 SPECT 成像重建流程，使用圆柱面探测器（32x64x4 晶体阵列）和极坐标系统矩阵。

## 整体工作流程

```
步骤1: 生成Factors文件 → 步骤2: 生成CntStat/List文件 → 步骤3: 图像重建 → 步骤4: 可视化与分析
```

### 步骤概览

| 步骤 | 目录 | 语言 | 说明 |
|------|------|------|------|
| 步骤1 | `Step1_GenerateFactors/` | MATLAB | 从系统矩阵文件生成极坐标Factors |
| 步骤2 | `Step2_GenerateCntStat/` | MATLAB | 生成正弦图和康普顿事件列表 |
| 步骤3 | `Step3_Reconstruction/` | Python | SC重建 或 JSCCSD重建 |
| 步骤4 | `Step4_Visualization/` | MATLAB | 图像可视化、CRC/CNR计算 |

---

## 环境要求

### MATLAB
- MATLAB R2022b 或更高版本
- 需要的工具箱：Image Processing Toolbox, Statistics and Machine Learning Toolbox

### Python
- Python 3.8+
- PyTorch 1.12+（建议 CUDA 版本）
- NumPy

安装 Python 依赖：
```bash
pip install torch numpy
```

---

## 详细操作步骤

### 步骤1：生成Factors文件

详见 `Step1_GenerateFactors/README.md`

1. 将数据文件放入 `Data/` 目录
2. 按顺序运行MATLAB脚本
3. 生成的Factors文件需要复制到 `Step3_Reconstruction/Factors/` 目录

### 步骤2：生成CntStat文件

详见 `Step2_GenerateCntStat/README.md`

1. 确保步骤1的Factors文件已准备好
2. 运行 MATLAB 脚本生成正弦图和事件列表

### 步骤3：图像重建

详见 `Step3_Reconstruction/README.md`

SC重建：
```bash
cd Step3_Reconstruction/SC_Recon
python main_local_cntstat.py --e0-list 0.511 --data-file-name ContrastPhantom_240_30 --count-level 1e9 --sc-iter 10000 --save-iter-step 100
```

JSCCSD重建：
```bash
cd Step3_Reconstruction/JSCCSD_Recon
python main_local_sparse_jsccsd_only.py --e0-list 0.511 --data-file-name ContrastPhantom_240_30 --count-level 1e9 --jsccsd-iter 5000 --save-iter-step 50
```

### 步骤4：可视化与参数计算

详见 `Step4_Visualization/README.md`

1. 将步骤3的重建结果复制到对应目录
2. 运行 MATLAB 可视化脚本

---

## 目录结构

```
Reproduction/
├── README.md                           # 本文件
├── Step1_GenerateFactors/              # 步骤1：生成Factors
│   ├── README.md
│   ├── Data/                           # 存放输入数据（SysMat.sysmat + CrystalMatrix.mat）
│   ├── genSysMatPolar_3D.m            # 核心脚本：直接从SysMat.sysmat生成极坐标Factors
│   └── PrintCrystalMatrix.m           # 辅助脚本：查看晶体矩阵内容
├── Step2_GenerateCntStat/              # 步骤2：生成CntStat
│   ├── README.md
│   └── GenProj_SPECT_PolarCoor.m
├── Step3_Reconstruction/               # 步骤3：图像重建
│   ├── README.md
│   ├── Factors/                        # 从步骤1复制
│   ├── CntStat/                        # 从步骤2复制
│   ├── List/                           # 从步骤2复制
│   ├── SC_Recon/
│   ├── JSCCSD_Recon/
│   ├── Figure_Dist_SC/
│   └── Fig_JSCC/
└── Step4_Visualization/                # 步骤4：可视化与分析
    ├── README.md
    ├── get_img_SC_Dist_PolarCoor.m
    ├── get_img_JSCCSD_Dist_PolarCoor.m
    ├── CNRCRC_SC_Dist.m
    ├── CNRCRC_JSCCSD_Dist.m
    ├── Figure_Dist_SC/
    └── Figure_Dist_JSCCSD/
```

## 注意事项

1. **数据准备**：步骤1需要 Monte Carlo 模拟生成的 SysMat.sysmat 文件和 CrystalMatrix 文件
2. **GPU推荐**：步骤3的重建过程强烈建议使用 GPU（CUDA）
3. **文件路径**：所有脚本使用相对路径，确保在正确目录下运行
4. **内存需求**：系统矩阵可能较大，确保有足够的内存