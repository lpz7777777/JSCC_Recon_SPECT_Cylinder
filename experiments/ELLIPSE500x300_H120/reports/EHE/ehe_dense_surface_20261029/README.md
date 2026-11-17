# EHE 218/440 keV表面积分加密

独立正式计算，保留74e129c4460163c5原生产版本。每个晶体可见面从16²=256点提高到256²=65536点；24个CPU代表点与512²参考比较通过选点门槛，但不是全矩阵物理认证。

仅A218、A440，2312bin、85×85×40原Cartesian网格、132040完整Polar点/78920活动单元、20view；原孔/晶体/能窗/材料参数和网格不变。PE未展宽和能窗矩阵重新生成；Scatter使用本块新PE未展宽输入按原公式重算，不拼接旧Scatter；每套生成完整Cartesian/Polar及自身S。原准直器散射均匀板深度近似保持，physical_calibration_passed=false，不把加密当作物理修复完成。C440→218不在本轮范围，不修改。

一个正式数组0–7，单元素1张5090、最多8张同时运行；实际GPU/独立CPU点值检查嵌在正式元素内部，无单独验证作业。最后完成的元素在同一正式作业内汇总两套矩阵；所有成功分块只读保留，不自动重投。无新输运、筛选或重建，无定时器。操作入口`ehe_dense_surface_workflow.py status/fetch`，原始数据在新generated目录，不覆盖旧结果。

正式数组 **1749085** 已登记，0–3为A218的4个轴向块，4–7为A440；每个块覆盖全部2312晶体及10个z层。输出`/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/ehe_dense_surface_20261029/formal_1749085`，最后一个成功块在同一作业内汇总两套Factors。尚未完成时不替换任何重建输入。

2026-10-29 18:07只读事实：原0–6共7块RUNNING、PE行进度约4.3%–5.5%，原7已FAILED/1:0，未进入GPU响应计算，目标`A440/slab_3`不存在。失败根因经实际`scontrol`证明：原7的numeric JobId恰为数组总编号1749085，查询总编号返回8条AllocTRES，旧解析器要求唯一记录而退出。不是OOM或计算核错误。

同日用户明确授权补投失败块。唯一补算正式任务 **1749513_7** 已提交（当前PENDING/Priority），1张5090、6CPU、自动内存、72小时调度时限；不重投原0–6。独立操作修复`9543c489bbac727a`只将资源查询改为显式数组元素并核对numeric JobId/ArrayJobId/ArrayTaskId，仍要求实际AllocTRES及原80%资源守卫。原`c8a802b2a9258fb5`源码、contract、参数、二进制和响应计算逐SHA不变，补算直接调用原冻结runtime，生成同一输出根下的缺失块，以便原汇总逻辑闭合。新allocation证据绑定修复适配器和manifest SHA，旧失败日志保留。无单独验证作业、定时器或新重建。

修复记录：`allocation_repair_freeze.json`、`allocation_repair_deployment.json`、`allocation_repair_submission_intent.json`、`allocation_repair_job.json`；本地6项真实多记录回归通过，登录节点SHA及实际数组身份检查通过，不当作正式GPU完成证明。提交入口`ehe_dense_surface_repair_workflow.py submit`存在登记/未决意图时禁止重复sbatch。原`ehe_dense_surface_workflow.py status/fetch`已记录原7失败与唯一补算7的有效作业映射，严格取回要求8个有效根全部COMPLETED/0:0。

实测估时见`runtime_forecast.json`：原7个运行块PE剩余线性外推约24.2–30.9小时；补算A440 slab3以同几何A218 slab3速度估计PE总耗时约31.8小时。加入速度波动和后续Scatter/两套Factors转换、SHA与I/O，规划约**补算开始后30–40小时**，排队时间另计。旧相同Scatter阶段实测167–784秒/块、旧三套转换约853秒只作后处理量级参考；本轮尚处首约5%晶体行，位置/节点共享可能改变速度，不能承诺完成时间，72小时是时限而非ETA。

18:10提交后复查：1749513_7仍PENDING/Priority，原7块全部RUNNING。Slurm `squeue --start`当前预测补算于**10月30日13:58:25**启动，属于可变化的调度估计；若按此启动，30–40小时规划对应**10月31日晚至11月1日清晨**完成计算。原始输出见`allocation_repair_post_submit_status.json`，不把预测启动或预测完成写成实际里程碑。

2026-10-30 12:15最新只读事实：8个有效分块均RUNNING，8个实际唯一5090 UUID；原7块已运行约19.6小时，补算1749513_7实际已于**10月29日19:14:51**启动（早于上述历史排队预测），已运行近17小时。218四块PE行数1472/1808/1836/1504，440四块1452/1808/1832/1304，各块总2312行，总PE进度70.37%；末块由补算提供。8块各12个嵌入点检查通过，当前Slurm/PE日志无新增错误，原失败1749085_7历史保留；尚无完整块receipt，未进入最终Factors汇总。

本次估时使用`live_timing_20261030_121548.json`全部真实时间线及最近1/2/4小时行速率。补算块是当前关键路径，PE剩余实测外推11.85–13.10小时；其他块约4.9–11.6小时。加入速度余量和后续Scatter/两套Factors及S/数值核对与I/O，规划从本次检查起约**12–16小时**，对应**10月31日0–4点左右**计算完成；不是完成保证，严格大文件取回及新重建不在该估时内。估时及SHA来源见本次`runtime_forecast_20261030_*.json`，旧10月29日估计保留，不再按旧排队预测推迟启动。
