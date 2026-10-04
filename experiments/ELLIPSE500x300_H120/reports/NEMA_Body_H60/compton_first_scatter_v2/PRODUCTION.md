# compton_first_scatter_v2运行簿

## 2026-10-04 实施与当前门控

本轮新建独立事件合同与配对实验，不恢复取消的1657887或旧自动任务。Geant4源码、物理模型和源宏部署至独立study目录，760生产worker固定3.17e9初级γ。

| 阶段 | 登记 | 实际状态 |
|---|---|---|
| 编译、16项分类器测试、三个同种子短程、首完整NEMA worker | maty15562315 | 通过 |
| 其余759worker | maty15562324 | 760个worker已完成，原32并发，空闲600CPU后提高至64 |
| 全量哈希/发射/身份、200NEMA旧分支字节回归、合并打包 | maty15562461，依赖afterok15562324 | 运行中，未宣称验收完成 |
| 65114五项数值测试与冻结分析代码 | analysis_deployment.json | 通过，未开始全量分析 |
| A/B独立S及空间效率验收 | validation_gate.json | 等待完整数据 |
| 能量域与边界离线诊断 | offline_summary.json | 等待完整数据 |
| 50次基线回归、A/B10次试跑、A/B2000 | imaging_job.json | 未提交，独立验证为硬门槛 |

完整NEMA首worker固定seed30093001、view1、5e6。PrimaryCount为218:1470218、440:3529782、其他0；旧List4998、新List3126，共同3115、仅旧1883、仅新11。旧List、两个CntStat及PrimaryCount与原已验收worker逐文件SHA完全一致。这里是Geant4原始List，尚未通过重建能量/层对/q筛选，不能等同最终接受率。

三个独立短程seed30093001/30093011/30093021分别运行legacy和paired，每次1e4；旧四输出全部字节一致。生产二进制SHA为`431c18339f7c9f46e7636e80b8962100229039450520d1d77835c5c37c988e15`。冻结输运部署、jobs清单和首worker完整输出证据见[deployment.json](deployment.json)、[transport_jobs.json](transport_jobs.json)与[configuration.json](configuration.json)。

数值测试在65114现有Torch环境运行：完整132040点q分块/rank不变、椭圆前后向内积≤1e-5、只读持久检查点保持原MLEM逐元素一致、边界积分体积Jacobian及B/ΔV不重复计体积、Geant4世界位置和晶体中心统一原点。项目原Geant4接口/计数分类测试11项通过；更新旧源码检查定位，避免读到历史注释。另有五项独立验收器测试，验证HOLD、错误S、视角不闭合、GPU/主存超80%、非有限历史和椭圆外泄漏都会拒绝。没有更改生产K*B核、能窗、展宽、网格或有效支持阈值。

### NEMA全部生产重放已完成

200个NEMA worker、20视角、1e9实际初级γ均完成。218/440/其他为293821153/706178847/0，与原生产一致；全部800份原List/CntStat/PrimaryCount逐文件字节一致，证据见[nema_replay.json](nema_replay.json)。原始旧List1093494条，理想List687890条，共同685297、仅旧408197、仅新2593。原始旧List包含218和440，两组原始总数不可直接解释为440接受效率损失；须以统一重建筛选后的事件身份统计比较。

输运依赖打包会逐worker验证所有旁路/原输出哈希，要求200个NEMA旧worker全部字节复现，实际总初级γ及各数据集数量闭合。分析随后计算NEMA原旧接受数97299、两组q筛选及各自S，独立圆/椭圆/九空间分区验收通过才冻结成像。HOLD必须定位，统计不足不追加超预算模拟。

当前没有新图像、CRC或尖峰改善结论。后续实际检查结果和资源记录应更新本运行簿，不把Slurm COMPLETED当作验收通过。
