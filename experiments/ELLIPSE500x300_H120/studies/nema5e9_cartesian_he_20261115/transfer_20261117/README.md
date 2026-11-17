# Cartesian矩阵传输加速（2026-11-17）

**传输已严格完成；正式作业已提交：** 北京时间2026-11-17 13:39三矩阵全部完整SHA闭合，记录`matrix_strict_transfer.json`（release `95022158ffedcdd7`）。A440/A218只读复用，串窗矩阵保留2,952,790,016B原前缀并补缺，其最终SHA `b922717276b136b5b66968ba9732c41bba780e5c221331f66304f1d1d1677c8a`匹配原冻结；科学字节不变。正式登记`formal_registration.json`：**1822462**，8×1普通5090，440 SC/440 JSCC/218 SC，10000/save50。13:40实际Slurm为PENDING、Reason=Priority、AllocTRES为空，无启动/迭代日志。没有科学失败，不重复提交。r6已进入FORMAL_WAIT；最新身份仍须核对PID/create_time/cmdline，后续以新鲜Slurm与完整退出/严格交付为准。下方过程记录全部保留。

**当前r6（北京时间2026-11-17 13:05启动）：** 唯一登记`completion_transfer_r6_registration_20261117.json`（PID18108，须核对create_time/cmdline），执行冻结`623264a88420a9cf`。A440/A218各21,285,888,000B已完整传输并SHA通过。r5在C440→218的2,952,790,016B处因源SSH超时留下一个空块，恢复时被原`0<prefix<bytes`拒绝，旧ACTIVE登记不能代表进程实际存活；r5已于12:48（北京时间）失败退出。

r6只改变空块续传边界：独占写锁确认目标是登记目录内的0字节文件且SHA为空文件SHA后，保留为同目录带时间戳的`.preserved`及fsync收据，再补传该范围。非空前缀仍两端SHA匹配后补缺，完整块仍SHA复用，大小/SHA异常仍停止；不覆盖任何成功数据。新`test_directory_r6.py`用合成小字节实际执行目录/合并/尾块/最终SHA和注入断线后的空块重试，网络为mock。最新检查`transfer_directory_checks_r6_lockaware_20261117.json`明确Windows未执行POSIX锁；它取代早期r6检查中不准确的Windows锁注释，旧收据保留。生产Linux路径使用实际flock，须由实际恢复收据确认。

准入`transfer_r6_recovery_authority_20261117.json`记录全部旧控制器退出、两份完整矩阵、串窗主前缀、3个完整未合并块和1个空块，以及没有正式作业/提交意图。新执行绑定`transfer_r6_execution_manifest_20261117.json`保留原18项及r5全部SHA身份；原有限截止日期与8×1普通5090正式三路不变。空块留证写入`transfer_empty_chunk_preservation_r6_20261117`，实际传输及提交仍以新鲜证据为准。

**历史r5（北京时间2026-11-17 11:59启动，12:48失败退出）：** 登记`completion_transfer_r5_registration_20261117.json`（PID18804），执行冻结`3e94cd63911ce79d`。r4已完整传输并独立SHA验收A440的21,285,888,000B，随后创建A218临时块目录时因父目录不存在而失败，实际进程已退出；r5后来完成A218，未发生正式提交。

r5在独立`chunked_transfer_r5.py`中仅把目标临时块目录创建改为`parents=True`，继承原长度帧、逐块有界重连、保留前缀、最终完整SHA和原Slurm衔接；r4源码、冻结、A440及所有日志保留不改。新增`test_directory_r5.py`在本地临时目录用合成小字节，实际执行三种通道的目录创建、顺序fsync合并、非整尾块、最终SHA/原子改名、成功文件复用和错误SHA拒绝；网络被mock，不冒充GPU或科学验收。检查记录`transfer_directory_checks_r5_20261117.json`，恢复准入`transfer_r5_recovery_authority_20261117.json`，新绑定`transfer_r5_execution_manifest_20261117.json`。原18项和r4全部绑定核对未改。新控制器不重复传输成功A440，只继续A218/C440→218；三者严格闭合后才唯一正式提交。

**历史r4（北京时间2026-11-17 05:14启动，05:37失败退出）：** 登记`completion_transfer_r4_registration_20261117.json`（PID36888），执行冻结`73c537f594a5b3a1`。r3已按身份停止，其A440至少8,518,139,904B连续前缀及全部分块保留。r4只修正watchdog关闭SSH后产生的EOF/退出码分类：自身超时关闭是可重试I/O，真实源断言/读取错误仍停止。原18项科学/执行绑定、矩阵、重建和Slurm配置均不改。

`transfer_framed_checks_r4_20261117.json`的9项本地检查通过：明确结束帧、两种无损传输、非零前缀、非法帧拒绝、有界重连、SHA错误停止、磁盘预留额度和watchdog分类。此次测试是本地合成字节协议检查，不是科学重建或GPU验证。r4的源错误与重连证据分别写入`transfer_source_errors_r4_20261117`、`transfer_retry_events_r4_20261117`。当前矩阵实际传输、完整SHA及formal_registration/Slurm才是后续提交依据，不能以控制器启动替代完成。

**历史r3：** r2在A440约6.37GB处遇到源SSH流提前结束（EOF）并停止；三套矩阵尚未完整传到GPU超算，正式重建尚未提交。已确认r2实际进程不存在，保留主前缀、3个完整未合并块及1个约130MB的未完成块。r3登记`completion_transfer_r3_registration_20261117.json`（PID35872，已停止），执行冻结`0c95e97f976d6a8b`。旧源码、日志和冻结保留。

r3每块最多5次有界I/O重连，采用2/4/8/16秒退避；每次从新连接重新校验保留前缀，只补缺失字节。接收器增加写锁，禁止旧连接收尾时并行写同一块。增加90秒无数据watchdog与源端退出码/stderr证据，避免再次只留下EOF而丢失诊断。数据SHA、源科学读取失败、权限/认证失败不自动当成临时网络问题重试；未知Slurm提交仍停止且不重复submit。

完整矩阵已传好的部分，经过目的端完整SHA核对后才计入原80GiB磁盘预留的已占用额度；不能把这部分再次当作剩余空间需求而卡住重连。原20%余量与正式资源限制保留。协议与重连策略7项本地检查通过，额外磁盘预留额度检查另有独立通过记录；扩展测试的重复旧receipt写入被不可覆盖守卫拒绝，原7项记录保留，未伪造覆盖。

r3证据：transfer_r3_recovery_authority、transfer_framed_checks_r3、transfer_storage_reserve_checks_r3、transfer_r3_execution_manifest_20261117；错误细节和重连记录在transfer_source_errors_r3_20261117与transfer_retry_events_r3_20261117。原科学release95022158ffedcdd7、矩阵字节、原8×1普通5090三路和截止期限保持。

**最新r2：** 原r1在A440约3.15GB处发生压缩流收尾等待，正式重建尚未提交。独立长度帧协议已修复，当前登记`completion_transfer_r2_registration_20261117.json`（启动PID2340，仍须核对create_time/cmdline），执行冻结`deed56e5e9ffb8ae`，当前指针仍为completion_current_controller_20261117.json。原r1已按身份停止，旧源码、manifest、日志、已合并前缀和全部未合并分块均保留。下文单连接/4路测速属于早期有限I/O证据，不能等同r1完整大文件成功。

北京时间11月17日04:39，r2先成功复用/补齐4个原块，再连续完成4个全新256MiB块；新增1GiB的实测区间112.50秒，约9.10MiB/s，A440累计达到5,296,914,432/21,285,888,000 B。区间速率不含旧chunk复用的虚高计数；progress中的启动累计速率包含复用，不能当成网络吞吐。证据`transfer_r2_verified_progress_20261117.json`。完整三矩阵传输及最终SHA仍未完成，不能提前宣称正式重建已提交。

## r2修复与保证

源端与接收端改用有长度上限的1MiB帧、明确的结束帧及独立SHA确认，不再等待管道EOF才处理尾部。源SSH的stdin设为DEVNULL、stderr用独立文件接收，并增加真实墙钟watchdog，避免读取阻塞绕过时限。

本地5项测试已通过，明确覆盖输入管道仍打开时接收端即可fsync/确认/退出、无压缩与无损压缩、非零前缀恢复、尾部以及非法帧拒绝。完整和部分旧chunk分别按原范围SHA或两端前缀SHA复用，只补缺失尾部。修复后的每个256MiB范围仍完整SHA核对，合并后仍按原21,285,888,000字节矩阵的冻结SHA独立检查。

新代码为framed_stream_r2.py、chunked_transfer_r2.py、finish_transfer_r2.py和test_framed_r2.py；旧r1代码不改。证据为transfer_r2_stop_20261117、transfer_framed_checks_r2_20261117、transfer_r2_execution_manifest_20261117。原科学冻结、三路正式配置及原有限截止日期不变，不取消任何超算计算任务。

用户授权：矩阵传输太慢，要求加快。三套实际矩阵已全部完成并正常退出，每套21,285,888,000 B，共63,857,664,000 B。原科学release95022158ffedcdd7、矩阵I/O补缺009433dd7e2b3b50及全部矩阵原字节保持。

## 实测与选择

| 路径 | 实测有效吞吐 |
|---|---:|
| 旧单路SFTP的连续进度观察 | 2.113 MiB/s |
| 单连接连续SSH流，真实48MiB切片 | 2.076 MiB/s |
| 单连接连续SSH＋zlib level1无损压缩 | 2.475 MiB/s |
| 4路独立连接＋相同无损压缩，实际192MiB | 8.241 MiB/s |

四路实测约为原单路3.90倍。单连接改协议本身没有提速，压缩样本仅缩小约11.4%，没有夸大为压缩解决瓶颈。样本覆盖同一真实冻结矩阵的起始、中部、末尾；各接收文件完整SHA、源/目的成功退出及fsync通过。该数字是有限I/O实测，不是完整63.9GB总耗时保证。记录见reports目录transfer_benchmark及transfer_parallel_benchmark_20261117.json。

## 当前执行与身份

原有限控制器2836及其唯一源SSH发送子进程9476按PID/create_time/cmdline核对后退出。没有取消超算计算任务，旧源码、登记、日志和原partial保留。停止intent和完成证明为transfer_ownership_stop_intent/transfer_ownership_stop_20261117.json。

唯一新有限控制器为generated中的completion_transfer_registration_20261117.json，启动PID11604，实际身份仍须核对。当前指针completion_current_controller_20261117.json。执行冻结1ddece02e8a2500d，manifest为reports/transfer_execution_manifest_20261117.json；原18项绑定与新增代码/测试/实测证明共同闭合。保留原2026-11-22T15:33:42.668790Z截止期限，不恢复聊天定时器。

新进程存活时，不并行执行submit/fetch/transfer，不修改其绑定源码。旧completion_repair登记中的ACTIVE只是历史，不代表原进程仍存活。

## 传输保证

- 首份A440原SFTP前缀先在源、目的两端计算SHA，逐字节一致才从其后续传，不清零或覆盖。
- 每份源矩阵按原完整SHA重新闭合，并生成256MiB范围的SHA计划。
- 最多4个范围并行，接收后逐范围SHA和退出检查，再严格按原顺序append至原partial；每次fsync并保存合并记录后，才清除本次派生范围临时文件。
- 除最终矩阵外，生产传输最多4×256MiB=1GiB范围临时数据；本地仅128KiB读取缓冲及有限socket缓冲，不落地64GB矩阵。I/O小样本文件另有独立目录，未混入正式输入。
- 最后独立重读完整目的矩阵SHA，必须等于原冻结SHA，才原子改名为正式目标。不能用压缩文件SHA替代解压后矩阵SHA。
- 不改变浮点精度、矩阵数值、物理核、网格、体积权重、匹配S、MLEM、检查点规则或原Slurm请求。

原handoff仅在新进程内替换transfer_one和矩阵状态提供函数；原文件不改。三矩阵全部严格传输后，继续唯一8节点×1普通5090的440 SC、440 JSCC、218 SC正式重建，各10000/save50；保留原未知提交intent拒绝盲重试和完整退出后strict fetch。

本地5项流/无损压缩/前缀恢复/错误拒绝检查及3项有序合并/尾块/越序拒绝检查通过。合成字节仅用于I/O回归，不冒充新物理验证。正式传输进度看reports/transfer_stream_progress_20261117.json及各通道transfer_complete_*；正式提交以formal_registration.json和实际调度器为准。
