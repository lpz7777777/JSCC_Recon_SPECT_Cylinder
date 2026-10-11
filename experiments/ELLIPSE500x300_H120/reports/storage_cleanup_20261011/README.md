# 2026-10-11 超算空间清理

定时任务已按用户要求停止：`jscc-5e10`实际PAUSED，现有定时任务均为PAUSED。原5090主1685272仍PENDING(Priority)，没有取消、修改或替换任何计算作业。本轮没有提交模拟、筛选、响应、验证或重建。

本轮实际移除429个SHA相同、具有保留本地备份的远端重复文件，共51211352294字节（47.69 GiB）。账号工作配额仍为800 GiB；df实际可用增加51211264000字节。上次2026-10-10的17.39 GiB清理不重复计入。

| 工作配额口径 | 清理前 | 清理后 |
|---|---:|---:|
| 已用 | 738.17 GiB | 690.48 GiB |
| 可用 | 61.83 GiB | 109.52 GiB |
| df使用率 | 93% | 87% |

| 实际移除范围 | 文件数 | 逻辑文件大小 |
|---|---:|---:|
| 旧Factors远端副本 | 291 | 36.85 GiB |
| 旧List远端副本 | 130 | 3.96 GiB |
| 重复传输归档 | 8 | 6.88 GiB |

完整本地和远端SHA逐文件比较，删除前再次核验保留备份及远端文件；每项unlink后写入flush/fsync删除日志。全部429条日志与验收收据一致，严格下载SHA通过；完整退出后再次确认所有删除路径不存在。

1205个缺少同大小本地副本或大小不同的历史文件继续保留。同大小但SHA不同的`Factors/511keV_RotateNum10_WrongPos/Sensi_d`、`Sensi_d_2`也保留，共1207个排除路径实际存在。没有删除唯一科学数据。

当前5e10输入、已验收筛选、标定矩阵、冻结发布、所有结果和日志，以及近期EHE完整响应、已安装conda环境、zxc和其他lpz工程保留。当前5e10与标定矩阵保护目录共1739项，路径/大小/mtime/mode清单摘要前后一致；删除目标与保护目录分离。旧Factors/List移出的远端副本需要时，可按`cleanup_acceptance.json`每项`backup_path`及SHA从本地恢复。

此次回收没有改变存储门槛：/ssd/scxi717实际仍root/0700，账号不可读写且目录配额仅1 GiB。800 GiB工作配额即使清空也容不下1530759784160字节的完整Compton缓存；原至少2 TiB可写持久SSD及实际I/O验证要求保持。独立实验仍未交付，validation10/formal10000尚未提交，定时推进继续停止。

原命令和容量见[inventory_before.json](inventory_before.json)与[inventory_after.json](inventory_after.json)；逐文件依据见[candidate_plan.json](candidate_plan.json)、[remote_proof.json](remote_proof.json)、[cleanup_acceptance.json](cleanup_acceptance.json)及[deletion_journal.jsonl](deletion_journal.jsonl)。定时停止见[automation_stop.json](automation_stop.json)，保留范围见[retention_summary.json](retention_summary.json)。实际执行代码身份见[executed_code_identity.json](executed_code_identity.json)，严格下载见[fetch_apply.json](fetch_apply.json)。

初次只读检查在作业Command解析处封闭失败，没有删除；改成精确key=value解析后继续，失败命令身份和诊断保留在execution_prove_rejected_guard及guard_parser_repair。
