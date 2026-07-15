# 225Ac Fr/Bi 双图耦合重建设计

本文记录 225Ac 场景下、218 keV 单光子通道受 440 keV 串扰时的算法开发方案。

## 当前实现状态（2026-07-15）

本地 CntStat-only 入口 `main_local_multi_energy_cntstat.py` 已实现第一阶段顺序估计：

```text
1. y440/A440 -> x440
2. b218 = C440to218*x440
3. y218 ~ Poisson(A218*x218 + b218) -> x218
4. xcombined = x440 + x218
```

其中 `b218` 作为固定加性背景进入 218 MLEM/OSEM 分母，不进入 A218 sensitivity，
也不对有噪声的 `y218` 做硬减法。该入口同时输出未校正的 218 图用于对照。

本文后续所述“同时更新 x_Fr 与 x_Bi”、440 List-mode 联合项和分布式版本仍是待实现
的完整耦合模式；不要把本地固定背景顺序估计误认为已经实现了双图联合迭代。

## 目标

当前 `distributed/python/main_dist_multi_energy.py` 和 `distributed/python/recon_osem_dist_multi_energy.py` 的多能量多输出链路仍然是**单图像模型**：每个任务把一个或多个能量通道的单光子/康普顿权重加到同一张 `img` 上。Type 5 的 `Image_J_S(440_218)keV_D440keV` 因此隐含“多个通道对应同一个空间分布”的近似。

对 225Ac 成像时，218 keV 主要来自 221Fr，440 keV 主要来自 213Bi。若 Fr 和 Bi 的空间分布不完全相同，218 能窗里仍会不可避免地包含 440 keV 光子因散射、能量展宽等造成的落窗事件。此时不应只用

```text
A_225Ac_218win_norm218 =
    A_218win<-218 + (Y440/Y218) * A_218win<-440
```

去重建一张图，因为这相当于假设 `x_Fr = x_Bi`。更合适的算法目标是同时估计两张图：

```text
x_Fr : 221Fr / 218 keV 对应的空间分布
x_Bi : 213Bi / 440 keV 对应的空间分布
```

并把 440 对 218 能窗的贡献显式写进 218 通道的 forward model。

## 响应矩阵与符号

建议后续代码中显式使用下列名字，避免把“能量”“能窗”“分支比”“串扰项”混在同一个参数里：

```text
A218       = A_218win<-218
             218 keV 发射光子进入 218 能窗的单光子响应。

C440to218  = A_218win<-440
             440 keV 发射光子经散射/能量展宽落入 218 能窗的单光子串扰响应。

A440       = A_440win<-440
             440 keV 发射光子进入 440 能窗的单光子响应。

D440       = 440 keV 康普顿/List 响应。
```

225Ac 相关的 gamma 发射概率采用项目当前约定：

```text
Y218 = 0.114
Y440 = 0.261
r    = Y440 / Y218 = 2.28947
```

若后续改用不同核数据库数值，应只改一个集中配置项，并在输出目录或日志中写明。

观测数据记为：

```text
y218  : 218 能窗单光子 CntStat
y440  : 440 能窗单光子 CntStat
L440  : 440 keV 康普顿/List 事件
```

## 前向模型

推荐在代码中优先使用“按 218 产额归一化”的形式，因为它和当前讨论的有效 218-window 矩阵写法一致：

```text
y218 / Y218 ~= A218 * x_Fr + r * C440to218 * x_Bi
y440 / Y440 ~= A440 * x_Bi
L440         ~= D440 * x_Bi
```

这里 `L440 ~= D440*x_Bi` 只表示 440 keV List-mode 事件似然的核由 `D440*x_Bi` 给出，不表示把每个事件逐条除以 `Y440`。对于 List-mode，事件列表本身保持原始事件；分支比/曝光尺度应在 `D440`、`Sensi_D440`、事件下采样或外部归一化中一致处理，并在日志里记录。

等价的绝对计数模型是：

```text
lambda218 = N_decay * (Y218 * A218 * x_Fr
                     + Y440 * C440to218 * x_Bi)

lambda440 = N_decay * (Y440 * A440 * x_Bi)

lambdaD440_event = N_decay * Y440 * D440_event * x_Bi
```

两种写法只能选一种贯穿实现。不要同时把观测数据除以 `Y218/Y440`，又把矩阵或 sensitivity 再乘一次分支比。

### 对已有 CntStat 的注意事项

如果 `CntStat` 来自真实 225Ac 发射流，或来自 Geant4 中按真实 225Ac 分支比在同一个 run 里发出的 218/440 keV 光子，那么观测计数本身已经包含分支比。此时：

```text
218-window mean count = Y218 * A218*x_Fr + Y440 * C440to218*x_Bi
440-window mean count = Y440 * A440*x_Bi
```

重建时可以使用绝对计数模型，也可以把 `y218` 除以 `Y218`、把 `y440` 除以 `Y440` 后使用归一化模型。关键是分支比只进入一次。

## Poisson 目标函数

忽略背景和正则项时，双图耦合模型可写为：

```text
min_{x_Fr >= 0, x_Bi >= 0}
    D_Poisson(y218_norm || A218*x_Fr + r*C440to218*x_Bi)
  + D_Poisson(y440_norm || A440*x_Bi)
  + D_ListMode(L440 || D440*x_Bi)
```

其中：

```text
y218_norm = y218 / Y218
y440_norm = y440 / Y440
```

`D_ListMode` 对应当前 JSCCSD 稀疏康普顿事件的形式：每个事件行 `D_i` 贡献 `log(D_i*x_Bi)` 型项，更新分子使用 `D_i^T / (D_i*x_Bi)`，灵敏度项使用与现有 `Sensi_d`/重算 sensitivity 一致的归一化方式。

如果需要加入背景、散射外部估计或随机项，应写成：

```text
lambda218 = A218*x_Fr + r*C440to218*x_Bi + b218
lambda440 = A440*x_Bi + b440
lambdaD   = D440*x_Bi + bD
```

背景项进入分母，但不进入待估图像的 sensitivity。

## EM/OSEM 更新式

以下公式使用按 218/440 产额归一化后的 binned 单光子数据。

218 能窗的耦合分母为：

```text
lambda218 = A218*x_Fr + r*C440to218*x_Bi
ratio218  = y218_norm / max(lambda218, eps)
```

440 单光子分母为：

```text
lambda440 = A440*x_Bi
ratio440  = y440_norm / max(lambda440, eps)
```

单光子部分的 backprojection 分子：

```text
num_Fr += A218^T * ratio218

num_Bi += r * C440to218^T * ratio218
num_Bi += A440^T * ratio440
```

康普顿/List 部分只更新 `x_Bi`：

```text
lambdaD_i = D440_i * x_Bi
num_Bi   += sum_i D440_i^T / max(lambdaD_i, eps)
```

对应 sensitivity：

```text
sens_Fr = A218^T * 1

sens_Bi = r * C440to218^T * 1
        + A440^T * 1
        + Sensi_D440
```

基本 EM 更新为：

```text
x_Fr <- x_Fr * num_Fr / max(sens_Fr, eps)
x_Bi <- x_Bi * num_Bi / max(sens_Bi, eps)
```

并保持非负和 NaN/Inf 防护，行为应与现有 `safe_em_update()` 一致。

### 与当前 JSCCSD 权重的关系

现有联合重建使用：

```text
SPECT weight  = alpha * get_weight_single(...)
Compton weight = (2 - alpha) * get_weight_compton_sparse(...)
s_map = alpha * Sensi_s + (2 - alpha) * Sensi_d
```

双图模式可以保留这个经验加权思想，但建议显式拆成三个可配置权重：

```text
w218_s : 218-window SPECT 权重
w440_s : 440-window SPECT 权重
w440_d : 440 keV Compton/List 权重
```

则更新式变为：

```text
num_Fr = w218_s * A218^T * ratio218
sens_Fr = w218_s * A218^T * 1

num_Bi = w218_s * r * C440to218^T * ratio218
       + w440_s * A440^T * ratio440
       + w440_d * D440^T * (1 / lambdaD)

sens_Bi = w218_s * r * C440to218^T * 1
        + w440_s * A440^T * 1
        + w440_d * Sensi_D440
```

若为了先快速复用当前参数，可设：

```text
w218_s = alpha
w440_s = alpha
w440_d = 2 - alpha
```

但长期更建议把三个权重写成独立 CLI 参数，因为 218 单光子、440 单光子和 440 康普顿承担的统计约束不同。

## 和现有单图像结果的关系

当前多能量 Type 5 输出：

```text
Image_J_S(440_218)keV_D440keV
```

是一张共享图像。它适合：

1. 多个能量通道确实对应同一空间分布；
2. 225Ac 局部平衡或点源标定，近似 `x_Fr = x_Bi`；
3. 只希望得到“225Ac 等效 218/440 联合图像”，而不区分 Fr 与 Bi。

未来 Fr/Bi 双图模式应至少输出：

```text
Image_Fr_Corrected
Image_Bi
```

建议额外输出便于对照的派生图：

```text
Image_Fr_plus_Bi
Image_Fr_BranchWeightedPlusBi = Y218*x_Fr + Y440*x_Bi
Image_218win_AcEffective     = 用 A218 + r*C440to218 单图近似得到的对照图
```

其中 `Image_Fr_plus_Bi` 和 `Image_Fr_BranchWeightedPlusBi` 的物理意义不同：前者是两个子体空间分布的直接相加，后者是按发射概率加权后的计数贡献尺度。

## 推荐实现路线

### 1. 先做本地原型

建议先新增本地原型，避免一开始就把分布式数据切分、all-reduce 和多任务调度一起引入：

```text
main_local_frbi_jsccsd.py
recon_osem_local_frbi_jsccsd.py
```

输入参数建议显式写出三个单光子响应目录和一个康普顿响应目录：

```text
--factor-218          Factors/218keV_RotateNum20
--factor-440          Factors/440keV_RotateNum20
--factor-440-to-218   Factors/440keV_to218win_RotateNum20
--cntstat-218         CntStat/218keV_RotateNum20/...
--cntstat-440         CntStat/440keV_RotateNum20/...
--list-440            List/440keV_RotateNum20/...
--yield-218           0.114
--yield-440           0.261
```

不要把 `440keV_to218win` 伪装成普通 `--e0-list 0.440` 的一个能量项。它不是独立观测通道，而是 218 能窗 forward model 中作用到 `x_Bi` 的交叉响应。

### 2. 再集成分布式任务

本地验证后，可在 `distributed/python` 中新增独立入口，或扩展现有 multi-energy 入口：

```text
distributed/python/main_dist_frbi_jsccsd.py
distributed/python/recon_osem_dist_frbi_jsccsd.py
distributed/scripts/jsccrecon_dist_frbi_jsccsd.sh
```

更推荐先独立入口，原因是 Fr/Bi 模式的数据结构和当前 `ReconTask` 的“一个任务输出一张图”不一致。等双图模式稳定后，再考虑把它抽象成新的 task type。

## 和现有代码的关键对接点

### 数据加载

需要同时加载：

```text
A218 local shard        -> 用于 218 分母和 x_Fr 分子
C440to218 local shard   -> 用于同一个 218 分母和 x_Bi 串扰分子
A440 local shard        -> 用于 440 单光子分母和 x_Bi 分子
D440 sparse/list data   -> 用于 440 康普顿分母和 x_Bi 分子
```

`A218` 和 `C440to218` 必须使用同一批 218 能窗 detector bins 做 subset，因为它们共同构成同一个 `lambda218`。如果二者行数或 bin 排列不一致，应直接报错，不应静默广播或截断。

### 旋转映射

三个 factor 目录的 `pixel_num`、`RotMat_full.csv`、`RotMatInv_full.csv`、`coor_polar_full.csv` 应一致。实现时建议显式校验：

```text
pixel_num(A218) == pixel_num(C440to218) == pixel_num(A440)
RotMat/RotMatInv shape 一致
coor_polar 坐标数一致
```

如果 `A218` 和 `C440to218` 使用相同几何网格，218 分母可写成：

```text
img_Fr_rotate = rotate(x_Fr)
img_Bi_rotate = rotate(x_Bi)
lambda218 = A218_subset * img_Fr_rotate
          + r * C440to218_subset * img_Bi_rotate
```

### 分布式 all-reduce

当前单图像分布式实现每个 rank 计算本地 detector-bin shard 的 `weight_local`，再对单张图做 `dist.all_reduce(weight_local, SUM)`。

双图模式需要两个本地权重：

```text
weight_Fr_local
weight_Bi_local
```

并分别 all-reduce：

```text
dist.all_reduce(weight_Fr_local, SUM)
dist.all_reduce(weight_Bi_local, SUM)
```

之后分别调用安全 EM 更新。

### sensitivity

需要维护两张图各自的 sensitivity：

```text
sensi_Fr = w218_s * Sensi_A218

sensi_Bi = w218_s * r * Sensi_C440to218
         + w440_s * Sensi_A440
         + w440_d * Sensi_D440
```

`Sensi_C440to218` 应来自 `Factors/440keV_to218win_RotateNum20/Sensi_s`，或由 `C440to218` 直接求和得到。它不能用 `A440` 的 sensitivity 代替。

### Compton sparse projector

`D440` 仍然只作用到 `x_Bi`。因此：

1. 只有 440 keV 需要构建 sparse projector 和加载 `sysmat_full`；
2. `C440to218` 不需要用于 Compton materialization；
3. `x_Bi` 的旋转图像传入现有 `get_weight_compton_sparse()` 即可；
4. 若复用当前 `Sensi_d` 重算逻辑，注意它目前还包含“把 Compton sensitivity 总和归一到 SPECT sensitivity 总和”的经验尺度，后续应在日志中明确记录。

## 伪代码

```python
for iter_idx in range(iter_num):
    for subset in subsets:
        num_fr = zeros_like(x_fr)
        num_bi = zeros_like(x_bi)

        for rotate_idx in range(rotate_num):
            x_fr_r = rotate(x_fr, rotate_idx)
            x_bi_r = rotate(x_bi, rotate_idx)

            # 218 window: coupled denominator
            A = A218_subset[rotate_idx]
            C = C440to218_subset[rotate_idx]
            y = y218_norm_subset[rotate_idx]

            lam218 = A @ x_fr_r + r * (C @ x_bi_r)
            ratio218 = y / clamp(lam218, eps)

            num_fr += inv_rotate(A.T @ ratio218, rotate_idx)
            num_bi += inv_rotate(r * (C.T @ ratio218), rotate_idx)

            # 440 SPECT: Bi only
            B = A440_subset[rotate_idx]
            y = y440_norm_subset[rotate_idx]

            lam440 = B @ x_bi_r
            ratio440 = y / clamp(lam440, eps)
            num_bi += inv_rotate(B.T @ ratio440, rotate_idx)

            # 440 Compton/List: Bi only
            for event_block in L440_subset[rotate_idx]:
                T = materialize_sparse_event_rows_to_fine(event_block, A440_full)
                lamD = T @ x_bi_r
                num_bi += inv_rotate(T.T @ (1 / clamp(lamD, eps)), rotate_idx)

        all_reduce(num_fr)
        all_reduce(num_bi)

        x_fr = safe_em_update(x_fr, num_fr, sensi_fr)
        x_bi = safe_em_update(x_bi, num_bi, sensi_bi)
```

实际实现时应加入 `w218_s/w440_s/w440_d` 权重、NaN/Inf 防护、空事件块处理、GPU/CPU 内存释放和保存中间迭代图。

## 验证用例

建议按以下顺序验证，避免一开始就在最复杂场景里判断算法是否正确。

1. **共定位 Fr/Bi phantom**

   设 `x_Fr = x_Bi`。双图重建的两张图应在形状上接近，并且单图像有效矩阵

   ```text
   A218 + r*C440to218
   ```

   的 218-window 结果应可作为对照。

2. **Bi-only phantom**

   设 `x_Fr = 0`、`x_Bi != 0`。218 能窗会有串扰计数，但双图模型应主要把它解释到 `x_Bi`，`Image_Fr_Corrected` 应显著低于单图像 218 重建结果。

3. **Fr-only phantom**

   设 `x_Fr != 0`、`x_Bi = 0`。440 单光子和 440 康普顿通道应接近空；218 通道应恢复 Fr 分布。

4. **Fr/Bi 不同位置 hot rods**

   设置 Fr 和 Bi 热区部分重叠、部分错位。对比：

   ```text
   Image_Fr_Corrected
   Image_Bi
   Image_218win_AcEffective
   Image_S_218keV
   ```

   重点观察 Bi-only 热区在 Fr 图中的残留。

5. **真实分支比混合数据**

   用 Geant4 按真实 225Ac 分支比混合发射 218/440 keV 光子生成 `CntStat` 和 `List`。重建时只在模型中使用一次 `Y218/Y440`，验证总计数尺度和 forward residual 是否合理。

## 建议评价指标

建议记录：

```text
CRC_Fr          : Fr 热区恢复系数
CRC_Bi          : Bi 热区恢复系数
CNR_Fr          : Fr 图对比噪声比
CNR_Bi          : Bi 图对比噪声比
Leak_Bi_to_Fr   : Bi-only ROI 在 Fr 图中的残留比例
Residual_218    : ||y218_norm - (A218*x_Fr + r*C*x_Bi)|| / ||y218_norm||
Residual_440S   : ||y440_norm - A440*x_Bi|| / ||y440_norm||
Residual_440D   : 440 Compton/List negative log-likelihood 或等价统计量
```

其中 `Leak_Bi_to_Fr` 是判断串扰校正是否有效的关键指标。

## 主要风险与限制

1. **可辨识性依赖 440 通道质量。** 如果 440 单光子和康普顿数据不足以约束 `x_Bi`，218 通道中的 `A218*x_Fr` 与 `r*C440to218*x_Bi` 会出现强耦合，Fr 图仍可能吸收部分 Bi 串扰。
2. **串扰矩阵必须物理可信。** `C440to218` 对能量分辨率、散射建模和能窗设置非常敏感。2026-07-09 已修复散射光子能量分辨率按 `1/sqrt(E)` 缩放的问题，旧散射矩阵不应再用于定量验证。
3. **不要重复分支比加权。** 数据生成、矩阵构造和 forward model 中只能有一个位置负责 `Y440/Y218`。
4. **当前 `intensity-list` 不是双图串扰接口。** 它会影响矩阵/灵敏度和下采样，语义上不同于 `yield_218/yield_440/cross_talk_scale`。后续实现应新增明确参数。
5. **有效 218-window 实测矩阵不能单独完成分离。** 如果只有 `A_225Ac_218win_norm218`，没有独立的 `A218` 和 `C440to218`，则无法在 Fr/Bi 空间分布不同的情况下做双图解耦。至少需要通过 Monte Carlo、额外标定或模型分解得到交叉响应。

## 首次开发检查清单

实现第一版时建议逐项确认：

- [ ] 三个 factor 目录的 `pixel_num` 和旋转映射一致。
- [ ] `A218` 与 `C440to218` 的 detector-bin 行顺序一致，并用同一 218 subset。
- [ ] `C440to218` 的 sensitivity 独立加载或独立计算。
- [ ] `Y218/Y440` 在日志里打印，并确认没有重复乘到数据和矩阵上。
- [ ] `x_Fr` 和 `x_Bi` 各自有独立初始化、更新、保存和迭代历史。
- [ ] 218 forward denominator 同时包含 `A218*x_Fr` 与 `r*C440to218*x_Bi`。
- [ ] 440 单光子和 440 康普顿只更新 `x_Bi`。
- [ ] 分布式版本分别 all-reduce `weight_Fr` 和 `weight_Bi`。
- [ ] 输出目录名包含 `FrBi`、`Y218`、`Y440`、`w218_s/w440_s/w440_d` 等关键配置。
- [ ] 至少通过共定位、Bi-only、Fr-only 三个最小 phantom 测试后，再跑复杂 Contrast Phantom。
