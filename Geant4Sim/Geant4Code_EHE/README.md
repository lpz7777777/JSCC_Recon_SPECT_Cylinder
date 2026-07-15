# EHE 平行孔 SPECT Geant4 模拟

本目录是一套独立于 `../Geant4Code/` 的 Geant4 工程，用于模拟与 MATLAB
`ConventionalSPECT` Params 一致的 EHE 三角晶格平行孔 SPECT。旧工程继续表示
10496-bin JSCC/GAGG 探测器；本工程只表示 1250 孔 Pb 准直器和 2312-bin NaI
探测器，不读取 `CrystalMatrix.txt`。

## 1. 对应的 Params

几何和事件能窗对应：

```text
Auxiliary_Studies/
  GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/
  FileGenerater_3D_Unified/
```

参数入口为：

```matlab
generate_ehe_pb_nai_218_440_response_params
```

对应三套响应：

```text
EHE_PbNaI_218keV
EHE_PbNaI_440keV
EHE_PbNaI_440keV_to_218keVwin
```

当前 Geant4 几何常数如下。

| 项目 | 数值 |
| --- | ---: |
| Pb 板尺寸 X/Y/Z | `330 / 50.5 / 165 mm` |
| 孔排布 | `25 x 50 = 1250` 个三角晶格圆孔 |
| 孔径 | `2.5 mm` |
| 最近邻中心距 | `5.9 mm` |
| 最近邻边缘铅隔厚度 | `3.4 mm` |
| NaI 阵列 | `68 (X) x 34 (Z) x 1 = 2312 bins` |
| 单个 NaI bin 尺寸 X/Y/Z | `4 x 10 x 4 mm` |
| NaI bin 中心距 X/Z | `4 / 4 mm`，无间隙 |
| 材料 | `G4_Pb`、`G4_SODIUM_IODIDE`、`G4_Galactic` |

这里的 `septal_thickness=3.4 mm` 按 Params 当前定义解释为相邻圆孔边缘之间的
最小 Pb 厚度，因此中心距为 `2.5+3.4=5.9 mm`。如果原始厂家资料中的 3.4 mm
实际表示中心距，则必须同时修改 MATLAB Params 和本工程，不能只改其中一侧。

## 2. 坐标对应

生产 macro 仍以以下位置作为 FOV 中心：

```text
(X,Y,Z) = (0,-245,0) mm
```

系统矩阵 Params 中，JSCC 局部 Y 原点距 FOV 中心 `170 mm`，第一层晶体中心还要
沿探测器方向增加 `30 mm`。由于第一层晶体厚 `3 mm`，其朝向 FOV 的前表面距
FOV 中心为 `170 + 30 - 3/2 = 198.5 mm`。EHE 准直器厚 `50.5 mm`，局部原点
位于 Pb 板中心，因此 EHE Params 使用
`fov2collimator0 = 198.5 + 50.5/2 = 223.75 mm`。转换到 Geant4 全局坐标后：

```text
FOV 中心             Y = -245.00 mm
Pb 板前表面          Y =  -46.50 mm
Pb 板中心            Y =  -21.25 mm
Pb 板后表面          Y =    4.00 mm
NaI 前表面           Y =    4.00 mm
NaI 中心             Y =    9.00 mm
NaI 后表面           Y =   14.00 mm
```

因此 Pb 前表面到 FOV 中心的距离为 `198.5 mm`，与 JSCC 第一层晶体前表面一致。
Pb 后表面和 NaI 前表面相接，间隙为 0。孔圆柱轴线和 NaI 厚度方向均沿全局 Y。

## 3. 探测器编号

copy number 和 CntStat 列均为 1-based 的 `1..2312`。循环顺序为：

```text
for x_index = 1..68
    for z_index = 1..34
        copy_number += 1
```

这与 MATLAB `build_detector.m` 生成并按列主序 reshape 后的最终
`Params_Detector.dat` 顺序一致。输出 CSV 的第 1 列对应 copy 1。EHE 工程不生成
Compton List。

## 4. 几何实现

`DetectorConstruction.cc` 执行以下操作：

1. 按 `build_collimator.m` 相同公式生成 1250 个孔中心并做均值居中；
2. 使用 `G4MultiUnion` 合并所有圆柱孔；
3. 从一块 `G4Box` Pb 板中一次扣除孔集合；
4. 放置 2312 个相邻的 NaI 计数单元；
5. 检查孔数、边界、最近邻中心距、铅隔厚度、NaI/Pb 接触面和 detector 数目；
6. 在当前运行目录导出实际采用的几何坐标。

导出的文件为：

```text
EHE_CollimatorHoles.csv
EHE_DetectorGeometry.csv
EHE_GeometrySummary.txt
```

几何文件在程序初始化时覆盖写入；计数文件仍按 run 追加。

## 5. 构建

要求支持 `G4MultiUnion` 的 Geant4 版本和 C++17 编译器。建议在独立 build 目录
构建：

```bash
cd Geant4Sim/Geant4Code_EHE
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

无图形环境的计算节点可关闭 UI/可视化：

```bash
cmake .. -DWITH_GEANT4_UIVIS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

可执行文件名为：

```text
ehe_spect
```

Windows + Visual Studio 2022 可使用：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DWITH_GEANT4_UIVIS=ON `
  -DGeant4_DIR="D:\Geant4-11.1\lib\cmake\Geant4"
cmake --build build --config Release -j
```

构建后以 `build\ehe_spect.exe` 为交互式启动入口。当前 CMake 会把 UI/Vis
驱动宏转换为工程自己的编译开关，并在 Windows 上强制使用 Geant4 Win32 UI；
无参数启动会自动执行 `vis.mac`。程序也会在交互模式下切换到 exe 所在目录，
因此从 Explorer 双击时仍能找到 `vis.mac`。

## 6. 冒烟测试

从新的空目录运行，避免 `CntStat_*.csv` 与旧结果追加混合：

```bash
mkdir -p run_smoke_218
cd run_smoke_218
/path/to/build/ehe_spect /path/to/build/smoke_218.mac > run.log 2>&1
```

`smoke_218.mac` 和 `smoke_440.mac` 使用沿 `+Y` 的窄束，穿过靠近中心且避开 NaI
像素边界的一个孔。每个 macro 发射 10000 个光子，用于快速检查：

- 初始化无 fatal exception；
- Pb/NaI 没有几何重叠警告；
- 输出恰好包含 2312 个 CntStat 数值；
- 主要计数集中在光束对应的 NaI bin；
- 218 窄束主要进入 `CntStat_218.csv`，440 窄束主要进入 `CntStat_440.csv`。

正式各向同性源的效率远低于该窄束测试，不能用 smoke 计数率估计系统灵敏度。

## 7. 与 Params 逐点核对

Geant4 至少初始化一次并生成几何 CSV 后，在运行目录执行：

```bash
python /path/to/Geant4Code_EHE/validate_against_params.py \
  --params-dir /path/to/FileGenerater_3D_Unified/output/EHE_PbNaI_218keV \
  --geant-output-dir .
```

工具直接读取二进制 `Params_Detector.dat`、`Params_Collimator.dat` 和
`Params_Image.dat`，应用与 Geant4 相同的 FOV 平移，然后按 ID 比较位置、尺寸和
孔半径。默认容差为 `1e-4 mm`。输出：

```text
EHE_GeometryComparison.txt
EHE_GeometryComparison.png   # 安装 matplotlib 时
```

预期结果为：

```text
status = PASS
detector_count = 2312
hole_count = 1250
minimum_septum_mm ~= 3.4
max_detector_error_mm < 1e-4
max_hole_error_mm < 1e-4
```

## 8. 源 macro

`example_225ac_point.mac` 是一个 FOV 中心点源示例。它使用项目统一的 gamma
产额：

```text
Y218 = 0.114
Y440 = 0.261
440/218 = 2.289473684211
```

`/gps/source/multiplevertex false` 保证每个 event 只选择一个源并发射一个 primary。
`/run/beamOn` 表示抽取的 primary gamma 总数，不是 225Ac 母体衰变数。

原 JSCC 使用的双能 phantom macro 也可以直接交给 `ehe_spect`，因为新工程刻意
保持了相同的 FOV 中心和世界坐标：

```bash
/path/to/build/ehe_spect \
  /path/to/Geant4Sim/Macro/ContrastPhantom_DualEnergy_10_30_240_30_225Ac/view01.mac
```

218 和 440 的空间分布仍可不同，整个 run 的两种能量权重仍由 macro 中所有 GPS
source 的总权重决定。

## 9. 输出与事件判定

每个 `/run/beamOn` 结束后追加写入：

```text
CntStat_218.csv   # 每个 run 一行，每行 2312 列
CntStat_440.csv   # 每个 run 一行，每行 2312 列
```

EHE 工程是纯单光子 CntStat 模拟，不判定康普顿事件，不保存 List 容器，也不生成
`List.csv`。`SteppingAction` 只按 NaI bin 累积沉积能量。

能量分辨率与 Params 一致采用：

```text
R(511 keV) = 13% FWHM
R(E) = 0.13 * sqrt(511/E)
```

因此自动光电峰窗为：

```text
218 keV: 196.30538 .. 239.69462 keV
440 keV: 409.17876 .. 470.82124 keV
```

每个 event 对各 NaI bin 的沉积能量独立做高斯展宽，随后遍历全部 2312 个 bin：
任一 bin 命中 218 或 440 能窗，就给该 bin 的相应 CntStat 加 1。一个 event 可以
给多个 bin 增加计数，不取最高能量 bin，也不提前返回。`SteppingAction` 使用
pre-step volume 归属沉积能量，并显式要求物理体名称为 `Scin`，防止把 Pb 或世界
体的 copy number 误当 detector ID。

## 10. 可视化

在带 Geant4 UI/Vis 的构建目录中不带参数启动：

```bash
./ehe_spect
```

程序自动执行 `vis.mac`，显示半透明 Pb 孔板、NaI 阵列和坐标轴。也可在 UI 中运行：

```text
/geometry/test/run
/control/execute smoke_218.mac
```

`/geometry/test/run` 比逐个 placement 启用 overlap check 更适合完整检查 2312 个相邻
NaI bin；这些 bin 共享边界但不发生体积重叠。

## 11. 当前物理边界

- 世界为真空，没有水、PMMA 或患者衰减体；GPS 的 Cylinder/Volume 只定义抽样
  分布，不会自动创建有材料的 phantom。
- 模型跟踪 gamma 和次级粒子在 Pb/NaI 中的相互作用，不模拟闪烁光、光电倍增、
  光导、电子学串扰、死时间和堆积。
- 当前为单线程 `G4RunManager`。多视角应以相互独立的进程和运行目录并行。
- 计数文件使用追加模式。重复运行前必须新建目录或明确管理旧输出。
- 默认随机种子取操作系统进程 ID（PID），因此同一台机器上同时运行的每个 worker
  都使用不同的随机数序列。为精确复现计算，或在多台机器间协调 seed，可为每个 worker
  显式设置不同的 `EHE_RANDOM_SEED`；程序启动时会打印实际 seed 及其来源。
- 在没有 Geant4 SDK 的工作站上，只能完成静态和 Params 数值验证；具备 SDK 的机器
  应执行构建、smoke、`/geometry/test/run` 和逐点对照工具。
