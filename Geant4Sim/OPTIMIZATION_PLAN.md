# Geant4 核医学成像模拟渐进优化方案

更新日期：2026-07-17

## 1. 范围与目标

本文针对以下两个独立应用制定优化路线：

- `Geant4Code/`：JSCC/GAGG 双能 SPECT，10496 个 detector bin，同时输出 218/440 keV CntStat 和双晶体 Compton `List.csv`。
- `Geant4Code_EHE/`：EHE Pb 平行孔/NaI SPECT，1250 个孔、2312 个 detector bin，只输出 218/440 keV CntStat。

目标不是单纯缩短某个 micro-benchmark，而是在不破坏几何、源分布、物理过程、能量展宽、能窗和输出含义的前提下，提高完整生产任务的吞吐量，降低启动、内存和 I/O 成本，并建立可重复的性能与物理回归体系。

当前生产模型的物理边界必须保持明确：GPS Cylinder/Volume 只定义源的抽样分布；两个工程目前没有启用水、PMMA 或患者实体，因此不包含物体内衰减和物体散射。当前也不模拟光学光子、光电转换、电子学串扰、死时间和堆积。以后增加这些物理内容时，本文中依赖“真空世界、单探测头”的剪枝或偏置方法必须重新论证。

## 2. 代码审阅结论

### 2.1 共同热点与风险

1. 两个入口都直接创建 `G4RunManager`，当前实际为单线程程序。虽然 `ActionInitialization::BuildForMaster()` 和 `Run::Merge()` 已存在，但没有被 MT run manager 使用。
2. `EventAction` 每个事件都清零并遍历完整 detector 数组。JSCC 每事件对 10496 个元素清零一次，并在结束时做展宽/能窗和 Compton 分类的多次完整扫描；EHE 每事件也固定清零并扫描 2312 个元素。绝大多数事件只触及零到少数 detector bin，这些访问主要是无效工作。
3. 两套代码都使用 `QBBC + G4EmStandardPhysics_option4`。对于当前 218/440 keV gamma 代理源，运行时主要是电磁输运；QBBC 中的强子部分是否有必要需要通过过程审计和物理回归决定，不能直接删除。
4. 当前输出使用追加模式。单进程时可用，但切换到 MT 后，现有 `EndOfRunAction()` 会在带 `PrimaryGeneratorAction` 的 worker 上写同名文件，而 master 的 `fPrimary` 为空。这会造成 worker 间文件竞争，必须在启用 MT 前修正。
5. 当前随机种子不适合作为统一的可复现实验协议。JSCC 使用秒级时间戳；EHE 默认使用 PID。应改成显式的 run/view/chunk seed，并把实际 seed 写入 manifest。
6. `G4GeneralParticleSource` 的非 flat 多源选择在 Geant4 11.4.2 中逐项扫描累计概率。项目当前的灵敏度点阵 macro 含 25600 个等权 source，而 `/gps/source/flatsampling` 尚未启用，平均每个事件会进行约 12800 次累计概率比较。这可能完全压过输运成本。
7. 两个应用分别复制了 Primary、Run、RunAction、EventAction 等代码。继续分别修改容易让物理判定、输出和并行修复发生漂移；性能稳定后应提取共享的小型核心库。

### 2.2 JSCC/GAGG 专有热点

- 几何包含前 30 层的 2304 个 GAGG 晶体，以及规则排列的 8192 个 `2 x 6 x 2 mm` 细晶体，总计 10496 个计分单元；钨块也作为大量 world daughter 放置。
- 这些 placement 当前直接作为 world 的同级 daughter，且大量 placement 在构造时启用了 overlap check。它会增加初始化成本，尤其不利于多进程分块运行。
- `SteppingAction` 每 step 获取 touchable、copy number、track ID 和过程名称。过程名称字符串比较只为 Compton List 分类服务，纯 CntStat 任务不应支付这部分成本。
- `Run::List` 在每个 worker 中动态增长，MT merge 时复制到 master。当前每视角数据量尚可，但更大 event 数或更多 List 接受事件会同时增加 worker 与 master 的峰值内存。
- 现有 1e9 和 1e10 双能生产结果可作为统计黄金参考。记录的总数分别为：

| 初级 gamma 总数 | CntStat 218 | CntStat 440 | List 行数 |
| ---: | ---: | ---: | ---: |
| `1e9` | 4,649,413 | 1,965,176 | 1,984,776 |
| `1e10` | 46,478,635 | 19,641,521 | 19,856,413 |

这些总数不能替代逐 bin、逐视角和 List 分布回归，但可以快速发现数量级错误。

### 2.3 EHE 专有热点

- Pb 板通过 `1250 cylinders -> G4MultiUnion -> G4SubtractionSolid` 建模。布尔实体内部导航可能是主要输运热点，需要用 profiler 和几何替代 A/B 测试确认。
- 2312 个 NaI 像素是相邻的直接 placement。它们可尝试参数化或 replica，但必须保持 copy number 与 `Params_Detector.dat` 完全一致。
- `SteppingAction` 每 step 通过物理体名称字符串 `"Scin"` 判断 NaI。应优先改成逻辑体指针比较。
- 几何构造时每次都执行 O(1250^2) 最近邻检查并写三个几何文件。这主要影响启动，不影响长 run 的稳态吞吐；可以保留在验证模式、在生产模式读取已验证的 geometry fingerprint 后跳过。
- 已有 `validate_against_params.py`、218/440 keV pencil-beam smoke macro 和明确的 `1e-4 mm` 几何容差，是几何优化的重要验收资产。

### 2.4 关键源码证据索引

| 结论 | 位置（以 2026-07-17 审阅版本为准） |
| --- | --- |
| JSCC 创建单线程 run manager | `Geant4Code/gamma01.cc:103` |
| EHE 创建单线程 run manager | `Geant4Code_EHE/ehe_spect.cc:91` |
| JSCC 每事件清零 10496 bins | `Geant4Code/src/EventAction.cc:98` |
| JSCC 展宽、CntStat、List 分别完整扫描 | `Geant4Code/src/EventAction.cc:123`、`:148`、`:163` |
| EHE 每事件清零并完整扫描 2312 bins | `Geant4Code_EHE/src/EventAction.cc:91`、`:112`、`:129` |
| MT 下现有输出条件落在 worker | `Geant4Code/src/RunAction.cc:92`、`Geant4Code_EHE/src/RunAction.cc:92` |
| EHE step hot path 比较物理体名称 | `Geant4Code_EHE/src/SteppingAction.cc:26` |
| JSCC 大量 placement 启用 overlap check | `Geant4Code/src/DetectorConstruction.cc:524`、`:565` |
| EHE O(N^2) 孔距校验及无条件几何导出 | `Geant4Code_EHE/src/DetectorConstruction.cc:98`、`:127`、`:187` |
| EHE `G4MultiUnion` 布尔孔几何 | `Geant4Code_EHE/src/DetectorConstruction.cc:201`、`:211` |
| Geant4 GPS 非 flat 线性概率搜索 | `Geant4_GPU/source/source/event/src/G4GeneralParticleSource.cc:204`、`:208` |
| `G4PhysicsVector` 可疑缓存上界 | `Geant4_GPU/source/source/global/management/src/G4PhysicsVector.cc:196`、`:200` |

实际生成的 25600 点等权 macro 当前有 281617 行、25599 条 `/gps/source/add`、没有 flat-sampling 命令，并以 `/run/beamOn 50000000` 结束。这说明 GPS 选择不是抽象的未来问题，而是现有工作负载中的可测热点。

## 3. 优化工作的硬规则

### 3.1 一次只改变一个主要因素

每个候选优化使用独立分支和独立构建目录。一次提交只改变一个主要因素，例如“启用 MT”“稀疏计分”“EHE 孔建模替代”，不得把 physics list、production cuts、编译器和几何同时改变。否则性能收益和物理偏差无法归因。

### 3.2 三类改动分开管理

| 等级 | 改动示例 | 验收要求 |
| --- | --- | --- |
| A：应保持语义 | Release/无 UI 构建、master-only 输出、指针比较、稀疏 touched-bin、参数查找算法改进 | 固定 seed 的顺序模式应逐 bin 相同；可保持 RNG 调用次序时要求 bitwise 相同 |
| B：改变执行/RNG 顺序 | MT/Tasking、chunk、NUMA、多进程、PGO | 不能要求 CSV 行顺序或 bitwise 相同；要求多 seed 统计等价和重建级等价 |
| C：改变物理估计器或模型 | production cuts、纯 EM 物理表、方向偏置、Russian roulette、GPU 输运 | 单独命名数据集；通过物理验证后才可替代基线，不能与 A/B 类结果混称 |

### 3.3 每一步都要有晋级门槛

候选优化只有同时满足以下条件才进入下一阶段：

- 代表性任务的 transport-only 和 end-to-end 吞吐均有稳定收益；默认要求中位数至少提升 10%，小于 10% 的改动只有在明显降低内存、启动成本或维护风险时保留。
- 至少重复 5 次，报告中位数、最小/最大值和变异系数；首次进程启动与稳态运行分开计时。
- 物理与输出回归通过。
- 峰值 RSS、输出大小和失败恢复能力没有不可接受的退化。
- 结果记录 Geant4 commit、应用 commit、编译器、CMake 配置、CPU/GPU、线程/进程布局、seed、macro 哈希和数据集哈希。

## 4. 阶段 0：冻结基准和物理合同

这是所有源代码优化的前置步骤。

### 4.1 建立基准目录

建议后续新增但本次尚未实现：

```text
Geant4Sim/perf/
  README.md
  benchmark.ps1
  workloads/
  baselines/
  results/<date>-<commit>/
  compare_counts.py
  compare_lists.py
  compare_spect_statistics.py
```

`benchmark.ps1` 应使用全新临时输出目录，禁止把基准追加到现有生产 CSV。每次运行生成 `run_manifest.json` 和 `timing.json`，失败时保留 log 但不把临时文件标记为有效结果。

### 4.2 固定工作负载矩阵

JSCC 至少包含：

| ID | 工作负载 | 目的 |
| --- | --- | --- |
| J1 | 218 keV pencil/point，10k 与 1M events | 快速正确性、单能能窗与启动时间 |
| J2 | 440 keV pencil/point，10k 与 1M events | 440 峰、向 218 窗散射 |
| J3 | 当前 8-source 双能 phantom 单视角，1M 与 10M events | 代表生产混合源和 CntStat |
| J4 | 能产生双晶体 Compton 的定向任务，至少获得 1e4 条 List | List 分类、顺序、内存和输出 |
| J5 | 25600-source 等权灵敏度点阵，先 100k/1M events | GPS source 选择与 macro 初始化热点 |

EHE 至少包含：

| ID | 工作负载 | 目的 |
| --- | --- | --- |
| E1 | `smoke_218.mac` | 孔中心透射、218 能窗、copy number |
| E2 | `smoke_440.mac` | 440 能窗与 NaI 响应 |
| E3 | `example_225ac_point.mac` 的 1M events | 各向同性双能端到端基准 |
| E4 | Pb 孔中心、孔边缘、铅隔中心的多位置/多角度 pencil beam | 准直器穿透、散射和几何替代回归 |
| E5 | 当前双能 phantom 单视角的 1M/10M events | 与 JSCC 使用相同源时的生产代表性 |

短任务主要用于回归，不用于外推生产吞吐。性能排名至少使用 1M events，并确保 transport 时间远大于初始化时间。

### 4.3 记录四段时间

每次运行至少分别记录：

1. 进程启动到 `runManager->Initialize()` 完成；
2. macro 解析和 GPS source 构建；
3. `BeamOn` 输运与事件计分；
4. Run merge、CSV 写出与进程退出。

同时记录 events/s、CPU 时间、CPU 利用率、峰值 RSS、major page fault、读写字节数、List 接受行数和每事件平均 touched-bin 数。只报告总 wall time 会掩盖 EHE 几何初始化、25600-source macro 解析或 List 写盘等完全不同的问题。

### 4.4 性能剖析

Windows 上优先使用采样 profiler，而不是在 hot loop 中加入大量计时输出。建议顺序：

1. Windows Performance Recorder/Analyzer 获取 CPU sampling、context switch、磁盘和 NUMA 线索；
2. Visual Studio CPU Usage 或 Intel VTune 定位函数级热点；
3. 只在确认需要时增加轻量计数器，例如总 step 数、NaI/GAGG 内 step 数、touched-bin 数、GPS source 数；
4. 对 EHE 分别统计 `G4SubtractionSolid/G4MultiUnion` 导航和电磁过程耗时占比。

### 4.5 物理回归协议

对于 A 类改动，顺序单线程、固定 seed、相同 macro 下：

- CntStat 两个文件要求逐元素完全一致；
- 可保持 RNG 调用顺序时，List 的 detector ID、flag 和 double bit pattern 完全一致；
- 若只改变 List merge/输出顺序，则先按事件诊断 ID和行内容 canonicalize 后再比较；生产 5 列 schema 不应因诊断需求被静默改变；
- EHE 几何 CSV 与 Params 最大误差继续小于 `1e-4 mm`。

对于 B/C 类改动，至少使用 10 个独立 seed，比较：

- 每视角与全局 218/440 总计数及其 Poisson/重复实验置信区间；
- 逐 bin standardized residual、卡方或似然比，并对多重比较做 FDR 控制；低计数 bin 不使用不稳定的百分比误差；
- 归一化投影的 NRMSE、相关系数、质心、FWHM/FWTM、均匀性和热点恢复系数；
- 440 keV primary 落入 218 keV 窗的 cross-window 比例；
- JSCC List 的接受率、两个 detector 的空间分布、两次沉积能量联合分布和 flag 分布；
- 使用实际重建流程比较图像 NRMSE、CRC、背景变异和收敛曲线。

推荐晋级标准是“优化版与基线版之差落在基线重复实验自身的 95% 波动范围内”，而不是给所有 bin 设一个任意的固定百分比。

## 5. 阶段 1：无物理风险的启动和配置优化

### 5.1 生产构建关闭 UI/Vis 和 verbose

- 使用 `Release`，设置 `WITH_GEANT4_UIVIS=OFF`。
- 生产 macro 设 `/control/verbose 0`、`/run/verbose 0`、`/event/verbose 0`、`/tracking/verbose 0`。
- 不在生产入口创建自定义 `SteppingVerbose`；debug/visualization 构建保留它。
- EHE 的三个几何导出文件和 O(N^2) 最近邻校验改为 `--validate-geometry` 模式；生产运行读取同一 geometry fingerprint。
- JSCC 停止每次打印完整 material table。
- 完整 `/geometry/test/run` 和 overlap check 放入 CI/几何发布步骤；生产构造中关闭每个 placement 的 `checkOverlaps=true`。

收益主要是启动时间，尤其适用于很多独立 chunk/worker。验收时必须先在验证构建中完成 overlap 和 Params 检查。

### 5.2 修正输出生命周期

- 输出路径、模式和 run ID 由 CLI 明确指定；默认拒绝覆盖或追加到非空结果目录。
- 先写临时文件，写完并 fsync/close 后原子重命名；manifest 最后标记 `complete=true`。
- CntStat 内部改成 64 位计数，避免长期累计和多 chunk 合并溢出。
- CSV 的列数、单位、1-based detector ID 和现有文件名保持兼容。

这一步主要提高可靠性，也为 MT 和 chunk 做准备。

### 5.3 等权 GPS 立即使用 flat sampling

对 25600 个 source 全部等权的灵敏度点阵 macro，增加：

```text
/gps/source/flatsampling true
```

Geant4 会从线性累计概率搜索切换到 O(1) 的整数索引抽样。只有所有 source 强度相同且计分不忽略非 1 的 primary weight 时才可认为与当前分布等价。当前点阵强度均为 1，理论上满足；仍需用 source 选择频数和 CntStat 多 seed 检验。

双能 phantom 的 8 个 source 权重不等，不能直接启用 flat sampling，因为当前 CntStat 没有按 primary weight 累积，启用后会改变物理结果。

## 6. 阶段 2：启用 CPU 多线程并建立混合调度

### 6.1 run manager 改造

- 使用 `G4RunManagerFactory::CreateRunManager()`，允许 CLI 选择 `Serial`、`MT` 或 `Tasking`。
- 增加 `--threads N`、`--seed S`、`--output-dir PATH`、`--run-id ID`。
- 只允许 master 写最终 CntStat 和合并后的 List；worker 只维护线程局部 `Run` 数据。
- `Run::Merge()` 继续逐 bin 合并，但改用 64 位计数并记录合并耗时。
- 检查 GPS 在 MT 初始化后的共享只读状态；禁止在 `BeamOn` 期间修改 source 配置。

特别注意：现有 `if (fPrimary && nbOfEvents)` 必须重构。MT 中 `fPrimary` 存在于 worker，而不是 master；沿用当前条件会导致多个 worker 同时追加同名 CSV。

### 6.2 随机数与可重复性

- 不再使用 `time(NULL)` 或 PID 作为正式运行 seed。
- 定义可审计的 seed 派生规则，例如 `hash(dataset_id, view_id, chunk_id, replica_id)`，并保存原始 base seed 和派生结果。
- 同一个 `(dataset, view, chunk)` 重跑得到同一结果；不同 chunk/worker 不得复用随机流。
- MT 与单线程结果不要求事件输出顺序一致，但要保留 event-level seed 或诊断 event ID，使问题事件可以重放。
- 不把线程调度顺序当作物理随机性来源。

### 6.3 本机线程矩阵

本机为双路 Xeon E5-2680 v4，共 28 个物理核、56 个逻辑处理器。至少测试：

```text
1, 7, 14, 28, 42, 56 threads
```

并分别测试 MT 与 Tasking，以及 Tasking 的 event/task grain。此前通用 Geant4 短基准中 14 线程优于 28/56 线程，但这不能代替本应用测试；SPECT 几何导航和计分的负载更重，最优点可能不同。

同时比较以下布局：

- 1 个进程 x 28 线程；
- 2 个进程 x 14 线程，每个进程绑定一个 NUMA node/socket；
- 4 个进程 x 7 线程；
- 20 个视角的受控队列，总 worker thread 数不超过已测最优 CPU 预算。

不能让每个视角都启动 28/56 线程，否则会严重 oversubscribe。

### 6.4 阶段 2 晋级条件

- master-only 输出无竞争、无重复行、无缺失行；
- JSCC CntStat 恰好 10496 列，EHE 恰好 2312 列；
- MT merge 前后计数守恒；
- 10-seed 统计回归通过；
- 选定线程数时 CPU 利用率、events/s 和 RSS 比单线程有明确收益；
- 56 个逻辑线程若比 28 个物理线程慢，则生产默认不得使用 56。

## 7. 阶段 3：稀疏事件计分，这是应用层首要热点

### 7.1 touched-bin 设计

每个 worker 保留：

```text
energy[detector_count]
touched_indices
```

首次向某 bin 加入正沉积能量时，把其索引加入 `touched_indices`；后续 step 只累加。事件结束时只处理 touched bin，处理完只清零这些元素。不得每事件 `fill` 全部 10496/2312 个元素。

为了让顺序单线程与旧实现保持 RNG 到 detector 的映射，事件结束前将 touched index 按升序排列，然后按升序执行高斯展宽。旧代码也是按 detector index 升序调用 `RandGauss`。这样可以在大幅减少扫描的同时维持随机调用次数和顺序。

### 7.2 合并事件末尾逻辑

在一次升序 touched-bin 遍历中完成：

1. 对正能量 bin 做高斯展宽；
2. 独立检查 218 和 440 窗，允许一个 event 给多个 bin 计数；
3. JSCC 同时收集超过 1 keV 的前两个 bin 和 multiplicity；
4. 保持现有 CntStat 与 Compton List 相互独立的语义；
5. 清零 touched bin。

不要为了速度改成“只取最大能量 bin”或“命中一个窗后 return”，这会改变当前已验证的多晶体计数语义。

### 7.3 step hot path

- EHE 用 `preStep->GetPhysicalVolume()->GetLogicalVolume() == detectorLV` 替代名称字符串比较。
- JSCC 暴露两个 scintillator logical-volume 指针，先做指针判断，再获取 copy number；不要把 world/W 的 copy number 偶然映射到 detector。
- `GetProcessDefinedStep()` 和过程名称只在 JSCC List 模式、primary track、正能量沉积且确有必要时读取。
- 增加明确的 `--scoring cntstat`、`--scoring cntstat+list` 模式。纯 CntStat 任务不创建 List 状态，也不做 Compton 分类。
- 比较手写 SteppingAction 与 `G4VSensitiveDetector` hit 累积；不要预设 SensitiveDetector 一定更快，保留 profiler 更优者。

### 7.4 验收

- 单线程固定 seed 下 CntStat bitwise 相同；
- 保持过程判断时 List 内容相同；
- touched-bin 排序前后 RNG 调用数一致；
- 记录 touched-bin 的 P50/P95/P99，证明稀疏假设成立；
- J3/E3 至少获得 10% 稳定 end-to-end 提升，否则检查输运是否已完全主导。

## 8. 阶段 4：List、分块和 I/O

### 8.1 List 内存策略

JSCC 的 CntStat 和 List 使用不同生命周期：CntStat 很小，可在 worker 内存中合并；List 可能增长到数百万行。依次评估：

1. 当前 worker vector + master merge，记录每视角峰值内存；
2. 每 worker 预估接受率后合理 `reserve`，避免多次扩容；
3. 超过内存阈值时，每 worker 分块写独立二进制临时文件，run 结束后按 worker/chunk 合并并转换为兼容的 5 列 CSV；
4. 为诊断版保留 event ID，生产版仍输出既有 5 列 schema。

不得让多个 worker 直接写同一个 `ofstream`，即使加 mutex 也会把事件处理串行化并产生不稳定行序。

### 8.2 生产 chunk

- 将每视角 5000 万或更大的 event 数拆成例如 100 万至 1000 万一块，实际大小由启动成本和失败恢复时间确定。
- 每 chunk 有唯一 seed、目录、manifest 和两张 CntStat；List 也单独保存。
- CntStat 逐 bin 求和；List 按视角合并行，不把 chunk 误当成新视角。
- 合并器验证事件数总和、列数、非负性、finite、detector ID 范围、List schema 和重复 chunk ID。
- chunk 完成前不得进入生产数据命名空间。

### 8.3 文件格式

CSV 保留为交换格式，但内部临时结果可比较：

- 定长 little-endian 64 位 CntStat；
- List 的结构化二进制块；
- 最终一次性格式化为 CSV。

GPU 或重建程序若能直接读取二进制/NPY/HDF5，可另外提供带 schema/version 的格式，但不能静默改变现有 MATLAB 消费路径。

## 9. 阶段 5：源生成优化

### 9.1 优先修复 Geant4 GPS 加权选择

Geant4 11.4.2 的 `G4GeneralParticleSource::GeneratePrimaryVertex()` 对非 flat 多源执行：

```cpp
while (rndm > sourceProbability[i]) ++i;
```

累计概率已单调排序，因此可在 Geant4 本地优化分支中改成 `std::lower_bound`，从 O(N) 降到 O(log N)，并保持“第一个累计概率 >= rndm”的边界语义。实施步骤：

1. 在 `G4GeneralParticleSourceData` 提供只读的 source-index 查找方法，避免暴露可变 vector；
2. 为随机数等于累计概率边界、首尾 source、1/2/8/25600 source 写单元测试；
3. 固定一组人工随机数，确认旧线性搜索与新查找返回完全相同的 index；
4. 用 J3 和 J5 分开测量。J3 只有 8 个 source，收益可能很小；J5 应明显下降；
5. 保留 flat equal-source 的 O(1) 快路径。

这个补丁比为每个应用复制一份 GPS 更通用，也较适合后续形成可上游提交的 Geant4 修复。

### 9.2 紧凑点阵生成器

若 25600 个 `G4SingleParticleSource` 的构建、内存或 macro 解析仍是瓶颈，再建立应用级 generator：

- 从紧凑坐标/权重文件一次读入 source 表；
- 等权点阵用一个均匀整数抽样；
- 任意权重使用 binary-search CDF 或 alias table；
- 直接设置一个 primary gamma 的位置、能量和各向同性方向；
- 把 source index 写入诊断统计，验证抽样频数。

这会改变随机数消耗顺序，属于 B 类改动；要求统计等价，不要求与 GPS bitwise 相同。

### 9.3 方差缩减，而不是伪装成普通加速

当前世界是真空且只有一个探测头。各向同性发射中，大量 gamma 根本不朝向探测器。在严格定义接受立体角并正确传播 statistical weight 的前提下，可以研究：

- 只向覆盖整个探测器/准直器的立体角采样方向；
- 对每个 primary 乘以 `Omega / 4pi` 权重；
- CntStat 改为累计 `sum(w)` 和 `sum(w^2)`，报告有效样本量和方差；
- 针对孔穿透和散射使用 importance splitting/Russian roulette。

这是 C 类估计器变更。当前整数 CntStat 和后续 Poisson 噪声含义会变化，因此必须输出新 schema 和 normalization metadata。以后加入水/患者实体后，原本不朝向探测器的光子可能散射回来，简单方向裁剪不再严格无偏，需要重新设计。

## 10. 阶段 6：几何与导航优化

### 10.1 先用 profiler 证明导航占比

只有当几何定位/安全距离/布尔实体函数在 CPU profile 中占显著比例时才进入此阶段。每个几何候选必须使用同一 physics list、cuts、seed 和 scoring 实现 A/B 测试。

### 10.2 JSCC 几何候选

按以下顺序逐个测试：

1. 关闭生产 overlap check，但在 CI 保留完整检查；
2. 对 world logical volume 扫描 `SetSmartless`，比较导航时间、初始化和内存；
3. 把规则的 8192 个末层细晶体改成 `G4PVParameterised`，保持 copy 2305..10496；
4. 把前 30 层晶体和钨块按类型参数化，坐标由预先验证的紧凑数组提供；
5. 只有 profiler 证明 world 同级 daughter 搜索仍昂贵时，再尝试按层建立真空 mother 层级。

验收必须导出全部 10496 个中心、尺寸、材料和 copy number，与现有 `Detector.csv`/Factors 顺序逐点比较。增加相同材料的 mother 边界也可能改变 step 划分，因此不仅要检查坐标，还要跑固定 seed 能量沉积回归。

### 10.3 EHE 几何候选

比较两个等价表达：

- 基线：`G4MultiUnion` 孔集合从 Pb box 中相减；
- 候选：完整 Pb box 作为 mother，在其中放置 1250 个参数化真空圆柱 daughter 表示孔。

后者可能让导航器更容易 voxelize，但不能假定一定更快。真空圆柱必须严格位于 mother 内，处理好几何容差和 Pb/NaI 接触面。

每个候选都必须通过：

- `validate_against_params.py`，最大坐标误差 `< 1e-4 mm`；
- `/geometry/test/run` 无 overlap；
- 孔中心、孔边缘、铅隔中心和多个斜角 pencil beam；
- 218/440 keV 的透射、Pb 散射、NaI 能谱和 cross-window 比例统计等价。

然后再尝试把 2312 个 NaI 像素改成 parameterisation/replica。copy number 顺序必须继续是 X 外循环、Z 内循环。

### 10.4 暂不默认使用 VecGeom/USolids

当前 Geant4 中该选项仍属于实验路径。只有标准 geometry 优化已经完成、现有布尔实体又确实是主热点时，才建立单独构建进行验证；不得把它作为第一阶段的生产依赖。

## 11. 阶段 7：编译、链接、PGO 与 NUMA

### 11.1 建立专用生产 Geant4 安装

在保留现有验证安装的同时，建立独立 production prefix：

- `Release`；
- Geant4 multithreading 开启；
- UI/Vis 可按部署需求关闭；
- `GEANT4_BUILD_VERBOSE_CODE=OFF`；
- `GEANT4_BUILD_STORE_TRAJECTORY=OFF`；
- 保留当前数据集版本和完整 build manifest。

关闭 verbose/trajectory 支持会减少不使用功能的成本，但必须保留一套 debug/visualization 安装供几何检查。

### 11.2 MSVC 候选开关

按独立变量测试：

- `/O2` 或当前 Release 优化；
- `/arch:AVX2`，本机 E5-2680 v4 支持 AVX2，但生成的二进制不再适合更老 CPU；
- `/GL` + `/LTCG`；
- 用 J3、J4、E3、E4 训练的 PGO。

Geant4 transport 具有大量分支和指针访问，AVX2/LTO 不一定显著；只保留实测收益。`/fp:fast` 不进入默认生产配置，因为它可改变浮点边界、能量比较和统计结果；若研究，按 C 类改动完整验证。

### 11.3 NUMA 与亲和性

Geant4 的 Windows worker affinity 路径不能作为可靠的双路绑核方案。优先由外部 launcher 把每个进程绑定到单一 processor set/NUMA node，并使用 14 个物理核；与单进程 28 核比较。

记录每种布局的：

- 每 socket CPU 利用率；
- remote NUMA memory、LLC miss 和 context switch；
- events/s、RSS 和功耗（可获取时）；
- 同时运行多个视角时的总吞吐，而不只看单进程速度。

生产目标是单位整机时间完成更多有效 events，不是让单个进程显示最多线程。

## 12. 阶段 8：physics list、production cuts 和输运剪枝

本阶段开始可能改变物理结果，所有输出必须使用新 physics configuration ID。

### 12.1 过程审计与纯 EM 配置

先列出 218/440 keV gamma、e-、e+ 在 Vacuum/Pb/W/GAGG/NaI 中实际注册和调用的过程，再建立“仅 EM option4”候选。对于当前直接发射 gamma、不使用 radioactive decay、能量低于常见 photonuclear 阈值的任务，强子过程可能没有运行时贡献，但仍需通过过程表和统计回归证明。

这项改动可能主要减少初始化和内存，而不是显著提高每 event 吞吐。若以后直接模拟 225Ac 衰变链、alpha 或更高能 gamma，必须恢复相应 decay/hadronic physics。

### 12.2 production cuts 扫描

分别为 Vacuum、Pb/W 和 detector 建立 region，逐级测试 e-/e+ range cut，例如：

```text
baseline, 1 um, 10 um, 100 um, 1 mm
```

具体有效能量阈值由材料转换表记录。gamma cut 不应随意提高，因为 Pb/W 特征 X 射线逃逸、低能散射和跨能窗污染对 SPECT 很重要。

验收重点：

- detector 总沉积能谱和光电峰；
- 相邻晶体能量共享；
- 440 -> 218 cross-window；
- Pb/W 穿透与特征 X 射线；
- JSCC Compton List 接受率和联合能谱；
- EHE 孔边缘/铅隔 pencil beam。

选择“满足物理容差的最大 cut”，不能只选择最快值。

### 12.3 物理模型不是编译优化

`G4EmStandardPhysics_option4`、Livermore 和 Penelope 的比较应作为模型系统学研究。更快的模型只有在满足所需 Rayleigh、photoelectric、atomic de-excitation 和材料能谱准确度时才能使用；不得因速度更快就称为等价优化。

## 13. 阶段 9：Geant4 内核局部优化候选

只有应用层热点、MT、GPS 和几何已处理后，才修改 Geant4 内核。优先候选：

1. GPS 累计概率从线性搜索改成 `lower_bound`，见阶段 5；
2. `G4PhysicsVector::FindBin(energy, idx)` 当前缓存条件同时比较 `energy >= binVector[idx]` 和 `energy <= binVector[idx]`，第二项很可能应为 `binVector[idx + 1]`。同文件的新 `CheckIndex()` 已使用 `idx + 1`。这会让缓存快路径几乎只在能量恰等于下界时命中。

第二项的实施要求：

- 先用 profiler/硬件计数证明 `G4PhysicsVector` 查找在 J3/E3 中显著；
- 写覆盖首 bin、末 bin、内部边界、恰等于边界、缓存命中/未命中的单元测试；
- 对旧 `GetBin()` 与修正缓存路径做全能量网格一致性比较；
- 固定 seed 做应用级 bitwise 回归；
- 单独记录 Geant4 patch commit，便于以后升级或向上游提交。

不要一开始就改 Geant4 的交叉截面、随机数或导航算法；应用中的单线程、稠密扫描和 GPS 线性搜索目前是更明确、更低风险的机会。

## 14. 阶段 10：GPU 路线

### 14.1 现实边界

当前 Geant4 11.4.2 没有可通过一个 CMake 开关启用的 CUDA/HIP/SYCL 通用输运内核。RTX 6000 Ada 不能自动加速现有 `G4RunManager`。把数千字节的单事件计分搬到 GPU 反而会被 PCIe 和 kernel launch 开销淹没。

因此 GPU 工作按以下顺序推进。

### 14.2 GPU 适合先做的部分

1. 继续把重建和系统矩阵运算放在 GPU；这通常比 event scoring 更适合批量并行。
2. 如果 List/投影后处理成为明显瓶颈，将大块二进制数据批量送入 GPU 做 histogram、能窗和重建预处理；必须与 CPU reference bitwise/数值比较。
3. 对 25600 点源的坐标/权重预处理可以 GPU 化，但其 CPU 开销相对数亿次输运通常很小，优先级低于 GPS O(N) 修复。

### 14.3 外部 GPU 输运原型

只有 CPU 路线完成后，再评估 AdePT、Celeritas、GGEMS 或专用 CUDA photon transport。先建立 EHE 的受限原型，因为其几何和输出更简单：Pb 平行孔、NaI detector、218/440 keV gamma、CntStat-only。

原型必须逐项确认支持：

- photoelectric、Compton、Rayleigh；
- Pb/NaI 的低能截面和 atomic de-excitation/荧光；
- 次级电子的处理以及局部沉积近似；
- 复杂孔几何和正确 detector copy mapping；
- 与当前能量展宽、双能窗和 cross-window 的接口。

GPU 原型使用独立数据集 ID，至少要求在 E1-E5 上物理统计等价，并且 end-to-end（含几何上传、数据传输、归并）相对优化后的 CPU 版本达到有意义的加速，建议门槛为 5 倍。否则维护两套物理实现的成本不合理。

JSCC 的钨/GAGG 三维不规则几何和 Compton List 更复杂，只在 EHE 原型成功后迁移。GPU 结果不能仅用总计数验证，必须比较能谱、空间投影、散射拓扑和最终重建。

## 15. 阶段 11：共享核心与长期维护

在主要行为稳定后，提取两个工程共享的组件：

```text
Geant4Sim/common/
  RunConfiguration
  SeedPolicy
  SparseEnergyScorer
  DualEnergyWindows
  CountRun
  OutputWriter
  RunManifest
```

两套 detector geometry、material 和 JSCC List classifier 仍保持独立。共享层只承载已经相同的行为，避免过度抽象。

CMake 改为显式 source 列表、C++17、可选 UI/Vis、可选 List、测试 target 和安装规则。增加自动测试：

- 稀疏/稠密计分相同；
- window 边界和负高斯截断；
- 一个 event 多 bin 计数；
- JSCC List 分类；
- MT Run merge；
- seed 派生无碰撞；
- CSV schema/列数；
- GPS source index 边界；
- EHE Params geometry validation。

## 16. 建议的实际执行顺序

严格按以下顺序推进，每一步通过门槛后再进入下一步：

1. 新建 perf harness，冻结 J1-J5、E1-E5、manifest 和物理回归脚本。
2. 对当前单线程 Release 做 5 次基准和 profiler，得到真实热点占比。
3. 关闭生产 verbose/Vis、重复 overlap check 和无条件 EHE 几何导出，分离启动与输运时间。
4. 给 25600 等权 source macro 启用 flat sampling；验证 source 频数和 CntStat。
5. 修复 master-only 输出、显式 seed 和 64 位计数，再启用 MT/Tasking。
6. 扫描 1/7/14/28/42/56 线程以及 1x28、2x14、4x7 混合布局，确定本机默认值。
7. 在两套代码实现升序 touched-bin 稀疏计分，先做单线程 bitwise 回归，再做 MT 回归。
8. 拆分 `cntstat` 与 `cntstat+list` hot path，完成 List 内存与 chunk/checkpoint 方案。
9. 修复 Geant4 GPS 非等权 source 的线性查找，并用 8-source 与 25600-source 两种负载验证。
10. profiler 若显示导航占主导，分别做 JSCC parameterisation/smartless 和 EHE 孔几何替代实验。
11. 建立 lean Geant4 production build，测试 AVX2、LTO、PGO 和 NUMA 绑定。
12. 只有物理团队批准后，研究纯 EM physics、region cuts 和无偏方向采样。
13. 只有优化后的 CPU 基线稳定后，启动 EHE GPU transport 可行性原型；成功后再考虑 JSCC。
14. 最后提取共享核心并把所有通过的基准/回归加入 CI。

## 17. 第一轮最值得实施的四项

按预计投入产出和风险排序：

1. **25600 等权 GPS flat sampling**：代码/宏改动小，直接消除每事件 O(25600) 线性选择，是特定点阵工作负载的最高优先级。
2. **MT/Tasking + 正确 master 输出 + 显式 seed**：两套程序目前都是单线程，这是生产总吞吐的最大通用机会。
3. **升序 touched-bin 稀疏计分**：消除 JSCC 每事件多次 10496-bin 扫描和 EHE 的 2312-bin 扫描，同时可以保持单线程 RNG 调用顺序。
4. **双路 NUMA 的 2 x 14 进程/线程布局实测**：本机不应默认使用 56 个逻辑线程；以整机 events/s 选择布局。

GPU、production cuts 和大规模几何重写都不应早于这四项。它们的工程和物理风险更高，而且若没有优化后的 CPU 基线，就无法判断是否真正值得。

## 18. 每次优化的结果记录模板

```text
change_id:
application: JSCC | EHE | shared | Geant4-core
change_class: A | B | C
application_commit:
geant4_commit:
compiler_and_flags:
physics_configuration:
geometry_fingerprint:
macro_sha256:
seed_set:
threads_processes_affinity:
events:
startup_s:
macro_parse_s:
transport_s:
merge_output_s:
events_per_s:
peak_rss_mb:
output_bytes:
physics_tests:
reconstruction_tests:
speedup_vs_baseline:
decision: promote | revise | reject
notes:
```

任何无法填完该模板的性能结论都只算探索结果，不能成为生产默认配置。
