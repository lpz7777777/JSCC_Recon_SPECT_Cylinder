# 源码索引

现存源码/宏/脚本共1778个。模块说明、顶层函数/类和行号见[source_index.json](source_index.json)。查询不导入这些模块。

| 职责 | 数量 | 分类入口 |
|---|---:|---|
| analysis-figures | 190 | [文件列表](code/analysis-figures/README.md) |
| calibration-physics | 34 | [文件列表](code/calibration-physics/README.md) |
| documentation-navigation | 14 | [文件列表](code/documentation-navigation/README.md) |
| execution-control | 71 | [文件列表](code/execution-control/README.md) |
| frozen-or-artifact-copy | 356 | [文件列表](code/frozen-or-artifact-copy/README.md) |
| geometry-source | 40 | [文件列表](code/geometry-source/README.md) |
| reconstruction | 152 | [文件列表](code/reconstruction/README.md) |
| shared-or-legacy-source | 745 | [文件列表](code/shared-or-legacy-source/README.md) |
| transport | 33 | [文件列表](code/transport/README.md) |
| verification-tests | 143 | [文件列表](code/verification-tests/README.md) |

```powershell
python tools/project_index.py files --group reconstruction
python tools/project_index.py find single_mlem
python tools/project_index.py files --group execution-control --topic ehe
```

[阅读代码路线](CODE_GUIDE.md)。`frozen-or-artifact-copy`为既有发布或图形副本；`retired-controller`保持停止。

[Python导入关系](source_dependencies.csv)通过source_id连接[source_index.json](source_index.json)中的完整路径。git_availability标明本地尚未提交文件，克隆后可能缺失；索引不声称它们已交付到Git。完全相同字节的[重复源码组](duplicate_source_groups.json)只用于追溯，不授权合并或删除。
