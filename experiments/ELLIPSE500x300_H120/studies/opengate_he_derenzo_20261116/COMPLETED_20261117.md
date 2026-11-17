# HE Derenzo完成与会议PPT交付

北京时间2026-11-17已完成四组独立HE Derenzo：218/440 keV，每种能量分别实际发射5e9或5e10个gamma。每组PyTomography MLEM200/save1，共800帧，全部严格取回和数值检查通过。科学冻结仍为34a52489b6362ba0；原HE响应46d438089710a6ce只读复用，未重算校准。

| 条件 | 成像数组 | 收集 | 本地重建 | 检查点 |
|---|---:|---:|---|---:|
| 5e9、218 keV | 16649504 | 16649585 | exit0，761.30秒 | 200 |
| 5e9、440 keV | 16649586 | 16649667 | exit0，361.39秒 | 200 |
| 5e10、218 keV | 16649668 | 16649669 | exit0，869.02秒 | 200 |
| 5e10、440 keV | 16650224 | 16650293 | exit0，493.42秒 | 200 |

各路权威：`reports/Derenzo/opengate_he_derenzo_20261116/<route>/strict_delivery.json`；完整输运成员、日志、资源、release及观察计数在同目录strict fetch记录中闭合。科学控制器已成功退出；旧registration中的ACTIVE是启动时记录，应以controller_exit.json及实际身份为准。不得重复模拟、响应、收集、重建或取回。

## PPT后处理补缺

原PPT后处理在output目录尚未创建时触发路径校验拒绝，保留旧候选、日志和失败exit。2026-11-17仅继续既有候选的最终打包，没有重跑科学计算。

Artifact Tool导入时未保留原嵌入工作簿，修复先逐条核对12张图表的全部轨迹数据一致，再逐字节恢复原chart XML、关系和原12个工作簿；没有生成新的图表数据或伪造来源。原candidate及补缺候选均保留。PowerShell5读取UTF-8无BOM中文文件名导致首次原生渲染找不到文件，改用ASCII脚本从实际output目录枚举精确候选，已成功在PowerPoint打开并渲染全部14页。

用户原文件已更新：

`D:/Research/0_Meeting/SingleMeeting/2026_H2/Meeting_LPZ_20261116_225Ac成像结果.pptx`

- 第11–14页替换为HE的5e9/5e10、218/440图，20/60/200次迭代；更新条件页与备注。
- 原NEMA8页、SC/JSCC图及12张可编辑CNR–BV图保持；Derenzo JSCC5e10仍以用户指定9950次为展示终点。
- 中心72mm MIP、完整XY、0–2统一相对热棒活度色阶、无滤波和亮度拟合。
- 最终文件2,879,841 B，SHA256：`d87ade3bd9392ef7c5fe4f0e067cb568793803010984009805bcbea4d01c5049`。
- 原始PPT备份保留在`Analysis_Plots/research/ac225_results_meeting_he_derenzo_20261116/source_backup`，原SHA为`64e25eb8f1c5bfd71b7eb65b97f915b568a997ee1066d177402fb4dbb71f86f5`。

最终证据：该图形目录下`build/validation_20261117.json`、`build/chart_preservation_20261117.json`、`build/scientific_package_qa.json`、`build/native_render.json`、`build/visual_qa_20261117.json`、`data/delivery_20261117.json`；研究reports目录另有`PPT_delivery_20261117.json`。

## 物理与显示限制

本次是现有HE深度插值、四孔相位平均CDR方案的Derenzo结果。有限准直器边缘与相位依赖近似未获物理认证。HE200次仍显示外围高值、条纹及噪声；原样展示，未据图像外观调整响应或筛选帧。HE4mm笛卡尔与原SC/JSCC柱坐标基线不同，不将相同迭代数当作同等收敛，也不以本次交付宣称分辨率认证通过。

本条仅结束HE Derenzo及其PPT替换。独立Cartesian NEMA研究继续；截至北京时间11月17日02:03，其A440/A218完成，C440→218完成11/15块，正式三路尚待矩阵齐备，由原存活控制器唯一衔接。
