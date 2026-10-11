# 2026-10-11 按用户指令删除zxc目录

用户明确要求删除 `/data/run01/scxi717/zxc`。本次实际递归移除该目录及其SCSPECT_EXP子目录，删除范围仅为这个授权路径。首次429个重复副本的清理收据保留不变，本次单独记录。

删除前检查确认：目标规范路径与授权路径完全相同，属于uid1721，目录及其父路径没有符号链接跳转，目标内没有嵌套挂载。六项本地控制器PID均不活跃；账户队列仅原5090主1685272，其Command保持原冻结selection.sh，脚本未引用zxc或SCSPECT_EXP。删除前保存并fsync独立intent，再调用Linux Python的shutil.rmtree；实际成功退出，删除后立即及独立再次检查均确认目标不存在。

目录删除前du实际占用226918653952字节（211.33 GiB）。JuiceFS账户配额显示在删除结束时仍在刷新；独立再次检查的UTC时间为2026-10-11T06:52:11.189080+00:00，df及statvfs一致。

| 800 GiB账户配额 | 删除前 | 配额刷新后的独立检查 |
|---|---:|---:|
| 已用 | 677.41 GiB | 465.96 GiB |
| 可用 | 122.59 GiB | 334.04 GiB |
| df使用率 | 85% | 59% |

账户可用净增加227039117312字节（211.45 GiB），最新可用358669889536字节（358.67 GB）。这是账户df净变化，与上述目录du用量分别记录，没有把上轮回收再次计入。

当前5e10及标定矩阵保护目录共1739项、19832548410字节，删除操作前后的路径/大小/mtime/mode摘要均为 `31936e784d7995a23a741908c9819bfbb15e26e1505821e77501a77cc68b4002`。原5090登记SHA仍为 `b593beff14a554d3e28a9219da26ea58748f1f58be1c2e3c7d028419b5c5efe1`。操作前后队列检查时1685272仍PENDING；随后独立检查发现它已由调度器自然进入RUNNING。没有取消、修改或重新提交作业，没有启动新重建。

八项现有定时任务均PAUSED。800 GiB配额未改变，完整Compton响应1530759784160字节仍超过这个配额；本次清理不解除既有STORAGE_HOLD或恢复定时推进。

[inspection.json](inspection.json)保存路径、挂载、占用及队列检查；[deletion_intent.json](deletion_intent.json)和[deletion_receipt.json](deletion_receipt.json)保存实际操作与前后证据。远端三份JSON已严格SHA下载，执行代码身份及SHA见execution_inspect.json、execution_apply.json；[verification_after.json](verification_after.json)保存配额刷新、目录不存在、作业自然状态和定时暂停核对。执行源码为仓库tools/research/delete_authorized_zxc.py，只允许inspect/apply这个固定授权路径；intent存在即拒绝重复apply。
