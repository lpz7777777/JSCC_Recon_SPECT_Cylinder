# 数据版本与目录保护

## 五层数据

1. **源码和配置**：Geant4Sim、Auxiliary_Studies、实验顶层及tools。可维护版本不等于某正式作业冻结版本。
2. **冻结与小型证据**：各研究reports中的合同、发布SHA、作业登记、strict delivery及QA；原始字节保留。
3. **运行数据**：各研究generated的投影、事件、系统矩阵、事件响应、完整历史与检查点。Git忽略，原路径不动。
4. **图表产物**：Analysis_Plots/research按日期保存figures/data/scripts及artifact SHA；既有产物保持原位置与科学身份。
5. **查询层**：docs/ac225及Analysis_Plots/results/current，按条件连接前四层，只存说明和指针，不复制原始数组。

## 完成与验收

提交、正常退出、严格取回、数值QA、视觉呈现和物理认证分开记录。完整远端历史检查与本地选帧检查分开；本地48选帧不能称为本地1600完整历史QA。失败父任务中成功的SC输出单独标记，不把整个任务伪装成完成，也不重复成功阶段。

目录名、旧checkpoint或partial文件不证明该研究已完成。优先读取原collection、run_manifest、completion、会计与strict delivery；索引只是入口。原稿/旧状态按原字节保存在整理审计的before目录。

## 版本命名

新研究使用独立日期/条件ID；矩阵版本显式区分原256点与dense65536、冻结C。真实Geant4和forward+Poisson分别归类。旧511keV、FOV120、未采用Derenzo设计和停止研究留作历史参考，不覆盖现用真值或自动复活。

## Git与空间

适合Git的小型文本为源码、配置、说明、解析几何、指标与SHA。矩阵、事件响应、大数组、完整历史、归档、PNG/PDF/PPTX等媒体、编译二进制和凭据保留在本地或原计算端，不加入本轮文本交付。

本轮没有移动/删除任何科学数据或发布，没有清理partial或取消任务。空间回收应另按明确路径、成功交付和依赖检查执行；不把“旧”或“失败”当作可删除理由。工程外SSH凭据不复制进目录盘点或报告。
