# 本地导航维护

从仓库根目录运行，Python 3.10+，仅标准库：

```powershell
python -B -X utf8 tools/navigation/build.py
python -B -X utf8 tools/navigation/check.py
python -B -X utf8 tools/project_index.py results --phantom NEMA --grid Cartesian4mm
```

`build.py`从人工审定的`docs/ac225/study_registry.json`和`result_registry.json`生成研究页、条件指针、CSV、网页、源码及浅层目录索引。它不联网、不导入科学脚本、不读取大数组、不更新AGENTS或原研究证据；刷新时间与证据观察时间分开。更改阶段前必须阅读冻结、交付与QA证据，不能由文件名自动推断。

`check.py`只读检查路径、链接、科学标签、阶段依据和受保护文件SHA。`audit_git.py`只审计暂存变更或指定出站范围，检查新增blob的1MiB上限、媒体、凭据与秘密模式。

旧刷新、建目录和验证命令兼容转入这里。20261103原实现保存在[legacy](../legacy/ac225_navigation_20261103/README.md)，其AGENTS写入和旧状态模板不再被调用。
