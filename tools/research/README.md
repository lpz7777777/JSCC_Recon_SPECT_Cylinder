# 专题研究工具

本目录保留已被研究/报告引用的分析、绘图、取回及控制适配脚本。程序名不代表只读；执行前读原研究合同与实现。

| 类别 | 推荐入口 / 副作用 |
|---|---|
| 当前导航维护 | [tools/navigation](../navigation/README.md)：离线构建、链接/标签/受保护SHA检查 |
| 查询 | [project_index](../project_index.py)：只读JSON，不导入科学脚本 |
| 图形与评价 | [研究清单](../../docs/ac225/EXPERIMENTS.md)、[结果](../../Analysis_Plots/results/REPORT.html)：按报告来源找实际脚本，通常写本地派生产物 |
| fetch/finalize/prepare/run类 | 可能传输、提交或修改远端状态；本次整理不调用 |
| 20261103导航实现 | 已[迁移归档](../legacy/ac225_navigation_20261103/README.md)，旧命令兼容新维护器，旧模板API关闭 |

按函数/职责查找：`python -B -X utf8 tools/project_index.py files --group analysis-figures`。原图集scripts副本及执行冻结可与工作区源码不同，均须按SHA追溯。
