# Geant4 双能 SPECT 蒙卡模拟代码

本文档说明 `Geant4Code/` 中 C++ 工程的用途、当前物理模型、探测器编号、输出格式，以及如何在另一台安装了 Geant4 的服务器上构建和运行。

文档对应 2026-07-10 的代码状态。当前工程主要面向 225Ac 成像方法开发，但还不是完整的 225Ac 放射性衰变链模拟。

## 1. 工程目标

当前链路用 Geant4 模拟双能 SPECT 探测过程：

1. MATLAB 脚本生成 20 个旋转视角的 GPS macro；
2. 每个 GPS event 从 218 keV 或 440 keV gamma 源中抽取一个初级光子；
3. Geant4 跟踪光子在探测器晶体和钨结构中的相互作用；
4. 对每个晶体的沉积能量施加高斯能量展宽；
5. 将命中 218/440 keV 能窗的单光子事件分别写入 `CntStat_218.csv` 和 `CntStat_440.csv`；
6. 将一部分两晶体康普顿事件写入 `List.csv`，供后续研究使用；
7. 将 20 个视角的 CntStat 按视角顺序合并后，与主工程的 218/440 keV Factors 配合重建。

当前明确不包含以下内容：

- 不直接发射 225Ac 离子，也不调用 Geant4 放射性衰变模块生成完整衰变链；
- 不模拟 alpha 粒子、衰变时间和子体迁移过程，而是直接给 218/440 keV gamma 设置不同的空间分布；
- 不模拟闪烁光、光电转换、电子学串扰、死时间和堆积；
- 当前 reconstruction 验证链路也暂不做串扰校正；
- 当前没有启用水或 PMMA 实体模体，因此没有物体内部的衰减和散射。

## 2. 目录与代码职责

```text
Geant4Code/
├── CMakeLists.txt
├── CrystalMatrix.txt
├── gamma01.cc
├── include/
├── src/
├── debug.mac
├── point.mac
└── vis.mac
```

| 文件或类 | 主要职责 |
| --- | --- |
| `gamma01.cc` | 创建随机数引擎、单线程 RunManager、探测器、物理表和用户 Action；执行 batch macro |
| `DetectorConstruction.*` | 读取 `CrystalMatrix.txt`，定义 GAGG 晶体和钨块并建立 10496 个探测器 bin |
| `DetectorMessenger.*` | 提供少量 `/testhadr/det/...` 几何和材料命令 |
| `PrimaryGeneratorAction.*` | 创建 `G4GeneralParticleSource`；实际生产源由 macro 中的 `/gps/...` 命令覆盖 |
| `SteppingAction.*` | 按晶体 copy number 累积每一步能量沉积，并记录康普顿过程信息 |
| `EventAction.*` | 每个 event 结束时做能量展宽、218/440 单光子能窗分类和康普顿 List 分类 |
| `Run.*` | 保存两个 10496-bin 计数数组和动态增长的 5 列 List |
| `RunAction.*` | 每个 `/run/beamOn` 结束后将 CntStat 和 List 追加到当前工作目录 |
| `ActionInitialization.*` | 注册 Primary、Run、Event 和 Stepping Action |
| `CrystalMatrix.txt` | `32 x 64 x 31 = 63488` 个探测器/钨块/空位标签，是运行时必需文件 |
| `CMakeLists.txt` | 构建 `gamma01`，并把 `CrystalMatrix.txt` 和示例 macro 复制到 build 目录 |

`GNUmakefile` 是历史文件。服务器部署优先使用 CMake，不建议把旧 Makefile 作为主构建方式。

## 3. 225Ac 双能代理源

### 3.1 当前源模型

生产 macro 位于：

```text
../Macro/ContrastPhantom_DualEnergy_10_30_240_30_225Ac/
```

它们由以下脚本生成：

```text
../ContrastPhantom_DualEnergy_Rotate_3D.m
```

当前采用两条 gamma 线作为 225Ac 成像代理：

| 能量 | 对应核素 | gamma 产额 |
| --- | --- | ---: |
| 218 keV | 221Fr | 0.114 |
| 440 keV | 213Bi | 0.261 |

218 keV 和 440 keV 的空间分布是有意设置为不同的：

- 218 keV：均匀背景加 rod 1、3、5 热区；
- 440 keV：均匀背景加 rod 2、4、6 热区；
- 每个热柱相对该能量自身的均匀背景为 6 倍活度。

这种错位用于近似表达 alpha 衰变后子体离开螯合药物，使 221Fr/213Bi 与母体药物分布发生偏移的情况。它不是源生成错误，也不应在整理 macro 时改成相同分布。

### 3.2 全 run 分支比

两张空间分布先分别归一化，再乘各自的 gamma 产额：

```text
q218(r) = 0.114 * x_Fr(r) / sum(x_Fr)
q440(r) = 0.261 * x_Bi(r) / sum(x_Bi)
```

因此约束的是整个 run 中 GPS 对两种初级 gamma 的期望抽样比例，而不是每个背景圆柱或每个热柱内部都保持 `0.114:0.261`。全 run 的期望份额为：

```text
218 keV: 0.114 / (0.114 + 0.261) = 30.4%
440 keV: 0.261 / (0.114 + 0.261) = 69.6%
440 / 218 = 2.289473684...
```

GPS 做随机抽样，所以实际整数发射数会有有限统计涨落。`/run/beamOn` 表示选取的初级 gamma 总数，不是 225Ac 母体衰变数。

当前参数为：

```text
视角数                 20
phi                    0, 18, ..., 342 deg
每视角 /run/beamOn     50,000,000
全部视角初级光子数     1,000,000,000
背景圆柱               直径 240 mm，高 30 mm
热柱直径               10, 14, 18, 22, 26, 30 mm
源中心                 (0, -245, 0) mm
```

macro 通过旋转热柱坐标表示不同投影视角；探测器几何本身保持不动。

### 3.3 源比例与探测计数比例不是一回事

`30.4%:69.6%` 只约束初级 gamma 的期望发射份额。最终 `CntStat_218.csv` 和 `CntStat_440.csv` 的总计数比例一般不会等于该比例，原因包括：

- 两个能量的几何探测效率不同；
- 在晶体和钨中的光电、康普顿过程不同；
- 能量展宽和能窗接受率不同；
- 440 keV 光子散射后可能落入 218 keV 能窗。

检查分支比时应检查 macro 的 GPS 权重或实际初级粒子统计，不能用两个 CntStat 的计数比直接反推源端权重是否正确。

## 4. 探测器几何与 10496 bins

### 4.1 CrystalMatrix 的解释

`CrystalMatrix.txt` 必须包含恰好：

```text
32 x 64 x 31 = 63488
```

个空白分隔的整数标签。允许的标签为：

| 标签 | 含义 |
| ---: | --- |
| `0` | 空位 |
| `1` | 闪烁晶体 |
| `2` | 钨块，不计入探测器 bin |

代码只按文本矩阵放置前 30 层。前 30 层必须恰好包含 2304 个标签为 `1` 的晶体。文本中的第 31 层标签不直接用于放置，而是被 C++ 中写死的细分末层完全替换：

```text
前 30 层晶体             2304
第 31 层细分晶体         128 x 64 = 8192
最终探测器 bin 数         2304 + 8192 = 10496
```

构造函数会检查文件长度、标签范围和前 30 层晶体数；几何放置结束后还会再次检查实际 copy 数是否为 10496。检查失败会以 `Detector001` 至 `Detector006` 的 fatal exception 终止。

### 4.2 几何参数和编号顺序

前 30 层晶体使用 `3 x 3 x 3 mm` 的 GAGG:Ce 晶体，x/z 中心间距为 4.2 mm，层方向 y 中心间距为 3 mm。末层使用 `2 x 6 x 2 mm` 的细分晶体，x/z 中心间距为 2.1 mm。

探测器 copy number 为 1-based：

1. copy `1..2304`：按 `layer -> j -> i` 循环扫描前 30 层，只给标签 `1` 的位置编号；
2. copy `2305..10496`：按 `j -> i` 扫描 C++ 细分末层；
3. CntStat 数组内部使用 0-based 下标，但输出第 1 列对应 copy 1；
4. `List.csv` 中的探测器编号也输出为 1-based。

该顺序已经与主工程下列文件的 10496 行探测器顺序核对一致：

```text
Factors/218keV_RotateNum20/Detector.csv
Factors/440keV_RotateNum20/Detector.csv
```

不要只用旧版、截断的 `CrystalMatrix.txt`。当前文件应有 63488 个标签；CMake 会将它复制到 build 根目录。

### 4.3 当前存在的实体

当前世界材料近似真空。实际启用的主要实体为：

- GAGG:Ce 闪烁晶体；
- `CrystalMatrix` 标签为 `2` 的钨结构；
- 真空世界。

`DetectorConstruction.cc` 中的外部钨屏蔽体以及水/PMMA Contrast Phantom 几何目前被注释。GPS 的 Cylinder/Volume 只定义初级粒子的空间抽样范围，不会自动创建有材料的圆柱实体。因此目前不包含模体内衰减和模体散射，只包含探测器及已启用钨结构中的相互作用。

## 5. 物理过程与事件分类

### 5.1 物理表

`gamma01.cc` 使用：

```text
QBBC
+ G4EmStandardPhysics_option4 替换默认电磁物理
```

当前使用普通 `G4RunManager`，是单线程程序。20 个视角的并行应通过多个独立进程完成，而不是期望一个 `gamma01` 进程自动使用多个 CPU 核。

### 5.2 能量展宽

`SteppingAction` 先按晶体累积真实沉积能量。`G4Step::GetTotalEnergyDeposit()` 表示该 step 内的局部总沉积能；沉积归属用 pre-step physical volume/copy number 判定，因为跨越几何边界时 post-step touchable 可能已经指向下一个体积。每个 event 结束时，`EventAction` 对每个有沉积的晶体独立施加高斯展宽：

```text
R(E) = 0.13 * sqrt(511 keV / E)
sigma(E) = R(E) * E / 2.35482
Emeasured = Edeposit + Gaussian(0, sigma)
```

负的展宽结果截断为 0。这里的 `0.13` 表示 511 keV 处 FWHM 能量分辨率为 13%。

### 5.3 单光子 CntStat

代码遍历 event 中全部晶体，并根据每个晶体各自的展宽后能量做 CntStat 分类。能窗定义为：

```text
window(E0) = E0 * (1 +/- R(E0)/2)
```

当前近似范围为：

| 输出 | 能窗 |
| --- | ---: |
| `CntStat_218.csv` | 196.305 至 239.695 keV |
| `CntStat_440.csv` | 409.179 至 470.821 keV |

分类规则为：

1. 某个晶体命中 440 keV 能窗，给该晶体的 440 CntStat 加 1；
2. 某个晶体命中 218 keV 能窗，给该晶体的 218 CntStat 加 1；
3. 继续遍历其余晶体，不只取最高能量晶体，也不提前结束 event；
4. CntStat 遍历结束后独立执行 Compton List 判定。

因此一个 event 可以给多个晶体的 CntStat 增加计数，也可以在贡献 CntStat 的同时
写入一行 List。分类不读取初级光子的能量标签；440 keV 光子散射后若任一晶体的
展宽能量落入 218 keV 能窗，会自然计入 `CntStat_218.csv`。

### 5.4 Compton List

当前 List 判定与 CntStat 判定互不排斥。即使 event 已有一个或多个晶体计入
218/440 CntStat，只要同时满足以下条件仍会写入 List：

- 恰有两个晶体的展宽能量大于 1 keV；
- 初级光子的第一次沉积晶体包含在这两个晶体中；
- 初级光子记录到一次 `compt` 过程，并进入第二个晶体。

当前接受的 List 行固定为 5 列：

```text
det1,energy1,det2,energy2,flag
```

| 列 | 含义 |
| --- | --- |
| `det1` | 第一相互作用晶体，1-based |
| `energy1` | 第一晶体的展宽后能量 |
| `det2` | 第二晶体，1-based |
| `energy2` | 第二晶体的展宽后能量 |
| `flag` | 当前接受事件固定为 `1` |

能量直接输出 Geant4 内部数值；Geant4 默认能量单位是 MeV，因此 218 keV 附近通常显示为约 `0.218`，而不是 `218`。List 不包含 event ID、视角 ID或初级能量标签。

`Run` 中的 List 现在使用动态 `std::vector<std::array<double,5>>`，不会再受旧版固定 1000 万行数组上限约束。但所有接受行仍保存在内存中，直到本视角的 `/run/beamOn` 完成后才统一写盘；大规模运行时需要监控内存。

## 6. 输出文件与追加行为

每次 macro 中的一次 `/run/beamOn` 完成后，程序在**进程当前工作目录**写入：

| 文件 | 当前状态 | 格式 |
| --- | --- | --- |
| `CntStat_218.csv` | 启用 | 每个 run 追加 1 行，每行 10496 个整数 |
| `CntStat_440.csv` | 启用 | 每个 run 追加 1 行，每行 10496 个整数 |
| `List.csv` | 启用 | 每个接受事件 1 行，固定 5 列；也可能为空 |
| `EnergySpectrum.csv` | 禁用 | `RunAction.cc` 中已注释 |
| `EventType.csv` | 禁用 | `RunAction.cc` 中已注释 |
| `Detector.csv` | 默认禁用 | `DetectorConstruction.cc` 中 `OutputDet=0` |

重要行为：

- 所有 CSV 都使用追加模式，不会自动清空旧结果；
- 重跑同一 macro 会增加 CntStat 行并重复追加 List；
- 输出没有 CSV 表头；
- 单个 run 中断时不会写出中间检查点，通常会丢失该视角尚未结束的结果；
- 多进程在同一目录写同名 CSV 会产生竞争和不可控混合；
- `List.csv` 没有视角 ID，因此即使顺序运行，也不建议把多视角 List 混在一个文件中。

生产运行推荐每个视角使用一个全新的独立工作目录。

### 6.1 已归档旧数据的状态

主工程中 `_Geant4JSCC` 命名空间下的 `1e9`、`1e10` Contrast Phantom 数据生成于
本次分类修复之前。当时只检查最高能量晶体，而且命中能窗后不再判断 List。这两组
数据仅保留用于问题复现和新旧逻辑对照，不能作为定量结果；部署当前代码后必须重新
模拟。详细统计见上一级目录的
`Geant4Data_ContrastPhantom_DualEnergy_225Ac_manifest.json`。

## 7. 服务器环境要求

建议环境：

- Linux x86-64；
- Geant4 11.x，原开发缓存曾使用 Geant4 11.0.2；
- 与 Geant4 匹配的 C++ 编译器；
- CMake 3.16 或更高版本；
- Bash；
- Python 3，仅用于结果检查和聚合，不参与 Geant4 模拟。

Geant4 的安装必须包含运行所需的数据集。通常执行安装目录下的 `geant4.sh` 会同时设置 `Geant4_DIR`、动态库路径和数据集环境变量。

需要传到服务器的最小内容为：

```text
Geant4Code/CMakeLists.txt
Geant4Code/gamma01.cc
Geant4Code/include/
Geant4Code/src/
Geant4Code/CrystalMatrix.txt
Geant4Sim/Macro/ContrastPhantom_DualEnergy_10_30_240_30_225Ac/
```

若还要在服务器上重新生成源 macro，再传输 `ContrastPhantom_DualEnergy_Rotate_3D.m` 及其依赖。

## 8. Linux 无界面构建

批处理服务器推荐关闭 UI 和可视化依赖：

```bash
source /path/to/geant4-install/bin/geant4.sh

cd /path/to/project/Geant4Sim/Geant4Code

cmake -S . -B build \
  -DWITH_GEANT4_UIVIS=OFF \
  -DGeant4_DIR=/path/to/geant4-install/lib/cmake/Geant4

cmake --build build -j"$(nproc)"
```

`Geant4_DIR` 必须指向包含 `Geant4Config.cmake` 的目录。如果 `geant4.sh` 已使 CMake 能找到 Geant4，可以省略 `-DGeant4_DIR=...`。

构建后应存在：

```text
build/gamma01
build/CrystalMatrix.txt
```

如需交互式 UI/可视化，改用 `-DWITH_GEANT4_UIVIS=ON`，同时保证 Geant4 安装包含相应驱动。生产 batch 不需要 UI/Vis。

## 9. Windows CMake 构建

在已配置 Geant4 和 Visual Studio 的 x64 Native Tools PowerShell 中：

```powershell
cd F:\path\to\Geant4Sim\Geant4Code

cmake -S . -B build `
  -G "Visual Studio 17 2022" -A x64 `
  -DWITH_GEANT4_UIVIS=OFF `
  -DGeant4_DIR=C:\path\to\geant4-install\lib\cmake\Geant4

cmake --build build --config Release -j
```

多配置生成器通常把可执行文件放在 `build\Release\gamma01.exe`，但 `CrystalMatrix.txt` 位于 `build\`。运行时应把工作目录设为含有 `CrystalMatrix.txt` 的目录，例如：

```powershell
cd build
.\Release\gamma01.exe C:\absolute\path\to\1.mac
```

### Windows 交互式启动

当构建时保持 `-DWITH_GEANT4_UIVIS=ON`，直接双击或运行 `build\gamma01.exe`
会启动 Geant4 的 Win32 命令窗口，并自动执行 `vis.mac` 打开 OpenGL 几何视图。
请使用 `build\gamma01.exe`，不要双击 `build\Release\gamma01.exe`；前者与
`CrystalMatrix.txt`、`vis.mac` 位于同一目录。关闭 Win32 命令窗口或在其中执行
`exit` 可结束交互会话。

传入 `.mac` 参数时仍是 batch 模式，结果写入启动进程的当前工作目录：

```powershell
cd F:\path\to\Geant4Sim\Geant4Code\build
.\gamma01.exe C:\absolute\path\to\1.mac
```

## 10. 单视角冒烟测试

不要直接用生产 macro 的 5000 万 event 做第一次测试。以下示例复制 `1.mac`，把 `/run/beamOn` 改为 10000，并在独立目录运行：

```bash
CODE_DIR=/path/to/project/Geant4Sim/Geant4Code
MACRO_DIR=/path/to/project/Geant4Sim/Macro/ContrastPhantom_DualEnergy_10_30_240_30_225Ac
RUN_DIR=/path/to/runs/smoke

mkdir -p "$RUN_DIR"
cp "$CODE_DIR/build/CrystalMatrix.txt" "$RUN_DIR/"
sed -E 's@^/run/beamOn[[:space:]]+[0-9]+@/run/beamOn 10000@' \
  "$MACRO_DIR/1.mac" > "$RUN_DIR/smoke.mac"

cd "$RUN_DIR"
"$CODE_DIR/build/gamma01" smoke.mac 2>&1 | tee run.log
```

日志中应出现类似信息：

```text
Loaded CrystalMatrix.txt: 32x64x31; using 2304 scintillators ... 8192 fine crystals.
Set Scin -- Done: 10496 scintillators
... write Csv File : CntStat_218.csv - done, accepted events = ...
... write Csv File : CntStat_440.csv - done, accepted events = ...
```

检查两个 CntStat 都恰好为 `1 x 10496`：

```bash
python3 - <<'PY'
import csv

for name in ("CntStat_218.csv", "CntStat_440.csv"):
    with open(name, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1, (name, "rows", len(rows))
    assert len(rows[0]) == 10496, (name, "columns", len(rows[0]))
    assert all(value.strip().isdigit() for value in rows[0])
    print(name, "shape=1x10496", "total=", sum(map(int, rows[0])))
PY
```

冒烟测试目录必须是新的或已确认没有旧 CSV。否则追加模式会使行数大于 1。

## 11. 20 视角生产运行

### 11.1 推荐目录结构

```text
runs/225Ac_dual_1e9/
├── view_01/
│   ├── CrystalMatrix.txt
│   ├── run.mac
│   ├── run.log
│   ├── CntStat_218.csv
│   ├── CntStat_440.csv
│   └── List.csv
├── view_02/
└── ...
```

每个目录只运行一次 `/run/beamOn`，所以每个 CntStat 文件应只有 1 行。视角编号 `01..20` 对应 macro `1.mac..20.mac`，合并时必须保持该数值顺序。

### 11.2 有限并发示例

`gamma01` 是单线程进程，可并行启动多个视角，但并发数应根据服务器 CPU 和内存确定。以下 Bash 示例最多同时运行 4 个视角：

```bash
#!/usr/bin/env bash

CODE_DIR=/path/to/project/Geant4Sim/Geant4Code
MACRO_DIR=/path/to/project/Geant4Sim/Macro/ContrastPhantom_DualEnergy_10_30_240_30_225Ac
RUN_ROOT=/path/to/runs/225Ac_dual_1e9
EXE="$CODE_DIR/build/gamma01"
MAX_JOBS=${MAX_JOBS:-4}

[[ -x "$EXE" ]] || { echo "missing executable: $EXE" >&2; exit 1; }
[[ -f "$CODE_DIR/build/CrystalMatrix.txt" ]] || { echo "missing CrystalMatrix.txt" >&2; exit 1; }

mkdir -p "$RUN_ROOT"
status=0
running=0

run_view() {
  local i="$1"
  local view
  view=$(printf "view_%02d" "$i")
  local dir="$RUN_ROOT/$view"

  mkdir -p "$dir" || return 1
  if [[ -e "$dir/run.log" || -e "$dir/CntStat_218.csv" || -e "$dir/CntStat_440.csv" || -e "$dir/List.csv" ]]; then
    echo "$view already contains a run log or output CSV; refusing to reuse it" >&2
    return 2
  fi

  cp "$CODE_DIR/build/CrystalMatrix.txt" "$dir/" || return 1
  cp "$MACRO_DIR/$i.mac" "$dir/run.mac" || return 1

  (
    cd "$dir" || exit 1
    "$EXE" run.mac > run.log 2>&1
  )
}

for i in $(seq 1 20); do
  run_view "$i" &
  running=$((running + 1))

  # gamma01 已混合高分辨率时间、PID 和 Slurm task ID；不需要错开启动。

  if (( running >= MAX_JOBS )); then
    wait -n || status=1
    running=$((running - 1))
  fi
done

while (( running > 0 )); do
  wait -n || status=1
  running=$((running - 1))
done

exit "$status"
```

不要让 20 个进程共享同一工作目录。若使用 Slurm/PBS，仍应让每个 array task 进入自己的 `view_XX` 目录后再执行同一个 `gamma01`。

### 11.3 随机种子

当前 `gamma01.cc` 默认把高分辨率时间、进程 ID、`SLURM_JOB_ID`、
`SLURM_ARRAY_JOB_ID` 和 `SLURM_ARRAY_TASK_ID` 混合成 Ranecu 合法种子，并在日志
打印完整初始化信息。并行 worker 可以同时启动，不再依赖 `sleep` 避免秒级重复。

需要严格复现时，可在每个任务中显式设置：

```bash
export JSCC_RANDOM_SEED=1234567
./gamma01 run.mac
```

显式值必须位于 `[1,2147483562]`。每个并行任务必须使用不同值，并保存日志中的
seed；设置非法值时程序会直接报错退出。

## 12. 结果检查与聚合

完成后，先检查每个视角都输出了恰好一行、10496 列，再按 `1..20` 的顺序聚合。以下脚本同时完成验证和聚合：

```bash
cd /path/to/runs/225Ac_dual_1e9

python3 - <<'PY'
import csv
from pathlib import Path

root = Path(".")
views = range(1, 21)

for energy in (218, 440):
    merged = []
    for i in views:
        path = root / f"view_{i:02d}" / f"CntStat_{energy}.csv"
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) != 1:
            raise RuntimeError(f"{path}: expected 1 row, got {len(rows)}")
        if len(rows[0]) != 10496:
            raise RuntimeError(f"{path}: expected 10496 columns, got {len(rows[0])}")
        row = [int(value) for value in rows[0]]
        if any(value < 0 for value in row):
            raise RuntimeError(f"{path}: negative count")
        merged.append(row)

    output = root / f"CntStat_{energy}_RotateNum20.csv"
    with output.open("w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(merged)
    print(output, "shape=20x10496", "total=", sum(map(sum, merged)))
PY
```

List 应保留每个视角一个文件，不要直接合并成无法区分视角的大文件：

```bash
mkdir -p List_by_view
for i in $(seq 1 20); do
  src=$(printf "view_%02d/List.csv" "$i")
  dst=$(printf "List_by_view/%02d.csv" "$i")
  cp "$src" "$dst"
done
```

还应检查：

- 20 个 `run.log` 都正常结束且没有 `G4Exception`、`Segmentation fault` 或 `Killed`；
- 两个聚合 CntStat 都是 `20 x 10496`；
- 每个视角的两个 CntStat 总计数均为非负且数量级合理；
- 不用探测后的 218/440 CntStat 比例强行验证源端分支比；
- 保留 macro、日志、Geant4 版本、编译选项和随机 seed 信息，保证结果可追踪。

## 13. 与主重建工程衔接

聚合后的数据分别对应：

```text
Factors/218keV_RotateNum20/
Factors/440keV_RotateNum20/
```

必须满足两个维度约束：

```text
行数 = 20 个旋转视角
列数 = 10496 个探测器 bin
```

推荐整理到主工程时保留能量和视角数：

```text
CntStat/218keV_RotateNum20/CntStat_<phantom>_<count>.csv
CntStat/440keV_RotateNum20/CntStat_<phantom>_<count>.csv
```

`1.mac..20.mac` 的顺序必须与 Factors 中的旋转矩阵顺序一致。不要按字符串字典序排列为 `1,10,11,...,2,...`；应按数值顺序 `1,2,...,20` 聚合。

当前 CntStat 可直接用于单光子 218 独立、440 独立以及 218+440 联合展示/重建链路。`List.csv` 是另一类康普顿候选数据，不应混入单光子 CntStat。

## 14. 常见问题

### CMake 找不到 Geant4

典型错误是找不到 `Geant4Config.cmake`。先执行：

```bash
source /path/to/geant4-install/bin/geant4.sh
find /path/to/geant4-install -name Geant4Config.cmake
```

然后将找到文件的父目录传给 `-DGeant4_DIR=...`。不要沿用另一台机器 CMake cache 中的绝对路径；换服务器后应新建 build 目录重新配置。

### 运行时报 `Cannot open CrystalMatrix.txt`

程序按当前工作目录读取相对路径 `CrystalMatrix.txt`，不是按可执行文件或 macro 所在目录查找。进入含该文件的运行目录后再启动，或把当前 `CrystalMatrix.txt` 复制到每个视角目录。

### 报 `Detector002`、`Detector004` 或 `Detector005`

说明矩阵被截断、含多余值，或不是当前标准的 `32 x 64 x 31` 文件。当前文件必须有 63488 个 `0/1/2` 标签，并使前 30 层含 2304 个晶体。

### CntStat 行数超过 1 或超过 20

输出是追加模式，说明目录中已有旧结果或同一个 macro 被重复运行。不要直接覆盖解释；换一个新的运行目录，并保留旧目录用于排查。

### 并行后 CSV 损坏或视角混合

多个进程使用了同一工作目录。每个视角必须有独立目录，结束后再由单进程按编号聚合。

### 无界面服务器出现 UI/Vis 相关错误

重新配置：

```bash
cmake -S . -B build -DWITH_GEANT4_UIVIS=OFF ...
```

无界面构建必须传入 macro 参数。直接运行不带参数的 `gamma01` 不会执行生产模拟；`debug.mac` 和 `vis.mac` 包含可视化命令，也不适合作为 headless 生产 macro。

### 进程被系统 `Killed`

先检查系统 OOM 日志并降低并发数。List 在一个 run 内保存在内存中，只有 run 结束时写盘；50,000,000 event 的实际内存需求取决于 List 接受数。

### 中断后没有可续跑的半个视角结果

当前每个视角只有一个很大的 `/run/beamOn`，并且在 EndOfRun 时统一输出，所以没有 event 级 checkpoint。只能重跑失败视角。若需要细粒度续跑，应把每个视角拆成多个独立、显式 seed 的 chunk，并在后处理时逐 bin 求和；不能把 chunk 当成新的旋转视角行直接拼接。

## 15. 修改入口

| 需要修改的内容 | 主要位置 | 注意事项 |
| --- | --- | --- |
| 218/440 空间分布、分支比、视角数和 event 数 | `../ContrastPhantom_DualEnergy_Rotate_3D.m` | 两个空间图必须分别归一化后再乘全 run 产额 |
| 探测器尺寸、材料、钨结构和末层细分 | `src/DetectorConstruction.cc` | 改后必须同步核对 10496-bin Factors 和 Detector 顺序 |
| 晶体空间标签 | `CrystalMatrix.txt` | 保持 63488 标签以及 C++ 读取顺序 |
| 物理表 | `gamma01.cc` | 当前为 QBBC + EM option4 |
| 能量分辨率、能窗和事件分类 | `src/EventAction.cc` | 当前能窗在构造函数中计算，修改后需重新编译 |
| step 级能量累积和过程判定 | `src/SteppingAction.cc` | copy number 到 CntStat 列的映射不能随意改变 |
| CSV 格式和文件名 | `src/RunAction.cc`、`src/Run.cc` | 下游脚本依赖 10496 列和 List 的 5 列格式 |
| 随机种子和 batch 参数 | `gamma01.cc` | 默认生成进程唯一 seed；可用 `JSCC_RANDOM_SEED` 显式复现 |

直接手工修改 20 个 `.mac` 容易破坏全 run 分支比。源模型改动应优先回到 MATLAB 生成脚本，重新计算权重并批量生成 macro。

## 16. 后续物理升级方向

要从当前“双 gamma 代理源”升级为更完整的 225Ac 蒙卡模拟，至少需要逐项评估：

1. 使用 225Ac ion 和 `G4RadioactiveDecayPhysics` 模拟完整衰变链；
2. 明确各子体的化学结合、迁移距离和时间分布模型；
3. 创建实际水/PMMA/组织材料模体，使源分布位于有材料的实体中；
4. 验证 218/440 keV gamma 产额、核数据版本和模拟统计定义；
5. 在批处理 manifest 中保存程序打印的每任务唯一随机 seed；
6. 设计 chunk 级 checkpoint 和集群任务合并；
7. 根据实验系统补充能量标定、死时间、电子学阈值以及必要的串扰模型；
8. 用点源、均匀源和已知几何逐级验证探测效率、能谱、空间编号和系统矩阵一致性。

在这些升级完成前，当前结果应表述为“按 225Ac 两条 gamma 产额加权、且允许 Fr/Bi 空间错位的 218+440 keV 代理源模拟”，不应表述为完整的 225Ac 衰变链模拟。

## 17. 当前验证状态

已经完成的静态和数据一致性检查包括：

- `CrystalMatrix.txt` 标签数为 63488；
- 前 30 层晶体数为 2304；
- C++ 末层晶体数为 8192；
- 最终探测器数为 10496；
- 按 C++ 循环重建的 10496 个坐标与 218/440 Factors 的 `Detector.csv` 一致；
- 20 个双能 macro 的全 run GPS 权重满足 `0.114:0.261`；
- `WITH_GEANT4_UIVIS=OFF` 的入口头文件已做条件编译。

当前本机没有可用的 Geant4 SDK 和 CMake 命令，无法在本机完成最新 C++ 版本的实际编译和运行。将工程传到服务器后，应先执行第 8 节构建、第 10 节冒烟测试，再启动 20 视角生产任务。
