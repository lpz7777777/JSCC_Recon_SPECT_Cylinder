# 1660254配对图集

见[完整科学验收报告](../ACCEPTANCE.md)。四条历史各40帧，原生统计完整120mm，展示MIP仅中央72mm。A旧定义/B理想首散射，共同固定A2000背景尺度，0–10灰度，无平滑。

| 文件 | 内容 |
|---|---|
| iterations_center.png | truth及100/500/1000/2000，中心z+1.5单层 |
| iterations_mip72.png | 同迭代中央72mm MIP；前两行Compton A/B，后两行JSCC A/B |
| final_planes_440_ComptonOnly.png | 2000次Compton真值/A/B轴、冠、矢、72mm MIP |
| final_planes_440_SinglePlusCompton.png | 2000次JSCC同布局 |
| final_multiplanar.png | 早期横向全通道总览；优先阅读上述两个较大图 |
| spike_noise_leakage_curves.png | 全40帧峰背景、CV、小单元质量、轴向泄漏 |
| crc_cnr_curves.png | 全40帧三个440热球CRC/CNR |
| density_integral_curves.png | 全40帧最大密度、背景、积分和高分位 |
| peak_location_curves.png | 全120mm最大值XYZ位置 |
| native_iteration_metrics.csv | 160行未裁剪原生指标 |
| sphere_iteration_metrics.csv | 480行三球指标 |
| comparison.json | 原始浮点指标、阈值结论、真值/历史哈希、共同尺度 |
| display_contract.json | MIP层/范围及固定尺度合同 |
| scripts/ | 本次绘图分析源码归档；复现用仓库实验根同名脚本 |
| artifact_manifest.json | 本目录全部交付文件SHA256，自身不递归哈希 |

复现：先使用已验收结果运行实验根compare_first_scatter.py；再运行plot_first_scatter_supplement.py --comparison本目录 --results generated/compton_first_scatter_v2/RemoteResults。两脚本会核对真值、历史及原有图集哈希，不修改重建。原历史及大数组保存在generated，按仓库规则不进入Git。
