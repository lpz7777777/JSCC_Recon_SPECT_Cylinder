# 查询与既有流程操作说明

## 先做无副作用查询

```powershell
python tools/project_index.py list
python tools/project_index.py list --topic nema
python tools/project_index.py list --topic derenzo
python tools/project_index.py show nema-jscc-5e10
python tools/project_index.py find 218
python tools/project_index.py files --group execution-control --topic derenzo
python tools/project_index.py dirs --topic derenzo
python tools/project_index.py check
```

查询命令只读本地catalog和文件存在性，不导入生产工作流、不连接SSH、不advance/fetch/submit。完整对象可在命令末尾加`--json`。目录索引重建命令：`python tools/research/build_project_catalog.py`，同样不运行科学计算；它更新本地证据快照和源码索引，不等同实时Slurm状态。

## 判断真实进度

1. `*_job.json`表示已登记，不等同RUNNING。
2. Slurm所有步骤COMPLETED/0:0才表示成功退出。
3. 独立verification通过后还需`*_acceptance.json`中的严格取回证明。
4. 图像/科学QA与最终报告须另行完成，不能以Git提交代表整个实验结束。

NEMA5e10状态见[当前快照](CURRENT_STATUS.md)。完整validation10已通过，但原正式启动gate的时限不够；不要反复调用旧release期待绕过时限，不能解除旧4090 STORAGE_HOLD或恢复失败/tmp方案。

## 控制器与并发

每次推进前核对该实验所有登记PID的实际进程身份。Windows会复用PID；已complete登记或另一个程序的新PID不代表旧控制器仍在运行。独立执行控制恢复入口为`experiments/ELLIPSE500x300_H120/resume_reconstruction_controllers_20261019.py`，仅用于此前授权范围，原14/21天截止保留。

Derenzo对应`generated/derenzo_equal_regions_mono_5e9_20261016/controller_registration.json`；NEMA对应`generated/jscc_geant4_5e10_10000/*registration.json`。活控制器旁不得并行advance/fetch/submit；不同实验只有既有人工授权的并行例外。定时任务保持暂停。

失败保留成功输出和原发布；实际sbatch意图不明时先核对调度器，不能重复提交。计算、取回或放行的任何变化继续遵守实验[AGENTS.md](../../experiments/ELLIPSE500x300_H120/AGENTS.md)和各自合同。本次目录整理没有扩大执行授权。

[远端访问规范](../REMOTE_COMPUTE_ACCESS.md)说明连接方式；密码、密钥和凭据仍位于工程外，不放进catalog或提交记录。
