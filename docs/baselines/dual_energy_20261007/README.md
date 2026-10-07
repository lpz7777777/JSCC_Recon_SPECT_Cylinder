# 固定基准登记：dual_energy_20261007

[流程与后续门槛](../../DUAL_ENERGY_BASELINE.md) · [研究结论回顾](../../DUAL_ENERGY_RESEARCH_REVIEW.md) · [本次整理验收](ACCEPTANCE.md)

本目录只存小型代码/输入/证明身份和清理记录，不存矩阵、事件块、重建数组或压缩包。

| 文件 | 用途 |
|---|---|
| manifest.json | 113个代码/配置源、31个冻结执行源、正式/短程/校准证明、74冻结发布文件、23原输入、27完整Factors及真实真值身份 |
| source_inventory.json | 538个相关代码条目的分类、源SHA、基准本地导入关系；含11个已删除旧脚本 |
| cleanup_manifest.json | 删除理由、原文件和Git blob SHA、清理前提交、恢复命令；旧README原字节保存位置 |
| validation.json | 本次CPU测试、只读完整字节检查、负向门控、文档/导入检查及其局限 |
| ACCEPTANCE.md | 人可读整理验收与交付边界 |

历史/兼容/诊断分类不授权执行作业。恢复旧脚本可用清单中的git restore命令；先前停止的试验和自动任务继续停止。

代码中31个已执行冻结源必须原字节一致。25个未冻结的Geant4/CUDA文本源及1个旧真值ROI元数据manifest记录LF规范SHA及Windows原字节SHA，仅允许平台行尾差异；其余执行证明保持原字节。源内容、算法、物理策略、几何、事件选择或S变化均需新版本，不覆盖本清单与既有证据。

运行默认核验：python -X utf8 tools/baseline/verify_dual_energy_baseline.py。可选完整本地输入/Factors读取使用--verify-data；可选冻结发布/原始结果读取使用--verify-payload --verify-results。默认跳过的大数据检查会明确说明，不能声称新增模拟或物理校准已经做过。
