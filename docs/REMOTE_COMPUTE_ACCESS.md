# 三处计算资源的安全连接与跨工程复用

核实日期：2026-09-24。此文件可供其他工程/协作者读取；它只记录连接方法、
身份和公开路径，不包含密码、私钥、解密后的凭据或访问令牌。
连接成功不等于授权修改其他工程。新工程应建立自己的远端目录、作业名和输出清单。

## 1. 资源分工

| 用途 | 目标与身份 | 认证 | 已验证环境 |
|---|---|---|---|
| GPU 重建 | `ssh.cn-zhongwei-1.paracloud.com:22`，用户名 `scxi717@BSCC-N56R5` | 账号密码；本机 DPAPI 加密存储 | 登录用户 scxi717、节点 ln01、Slurm、gpu_5090/gpu_4090 |
| Geant4 | `192.168.11.1:22`，用户名 maty | 加密 RSA 私钥，经 Windows ssh-agent 解锁 | 登录节点 ibcln01、Slurm cnmix、Geant4 11.1.0 |
| 自有服务器 | SSH 别名 `65114_lipeize` → `192.168.1.24:22`，用户名 lipeize | 本机既有 SSH 私钥和证书 | medphy-SYS-420GP-TNR、6×RTX A6000、CUDA12.8、MATLAB R2025b |

当前本地工程根目录：
`D:\JSCC_Recon\20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor`。
其他工程只需引用本说明和已审核的连接脚本，不需要复制任何密钥或密码文件。
通用入口副本放在 `C:\Users\Lipeize\.codex\REMOTE_COMPUTE_ACCESS.md`；仓库内本页为可版本追踪的来源。

## 2. scxi717：账号密码连接

交互式命令（普通本机 PowerShell，密码仅在终端提示中输入）：

```powershell
ssh -p 22 -l "scxi717@BSCC-N56R5" ssh.cn-zhongwei-1.paracloud.com
```

用户名包含 `@`，用 `-l` 明确指定。不要误用本机旧配置的
`scb7788@BSCC-A6` 或 2222 端口。账号密码不由 ssh-agent 缓存，其他终端也不会继承已登录会话。

### 供脚本使用的本机凭据

已审核入口：`experiments/FOV120/save_reconstruction_credential.ps1`。
仅首次设置或改密码时运行；它通过 `Read-Host -AsSecureString` 输入，
`Export-Clixml` 使用 Windows DPAPI 加密，保存到：

`%USERPROFILE%\.ssh\fov120_paracloud.credential.xml`

文件 ACL 限定当前 Windows 用户；**加密文件也不进 Git、不上传服务器、不发到聊天中**。
通常仅原 Windows 用户和机器能解密。管理员/已控制该用户进程的程序仍可读取其凭据，
因此这不是对本机恶意程序的隔离机制。不要自动读取无关账号凭据。

从任意工程使用（可按实际安装路径替换 Python）：

```powershell
$repo = 'D:\JSCC_Recon\20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor'
$python = 'C:\Users\Lipeize\AppData\Local\Programs\Python\Python312\python.exe'
& $python "$repo\experiments\FOV120\reconstruction_ssh.py" --command 'hostname; whoami; pwd'
```

依赖 Paramiko。脚本从已存在的 OpenSSH `known_hosts` 校验主机密钥，未知或改变的
主机密钥会被拒绝。首次指纹应经可信渠道核对；不要禁用校验或无条件接受变更。
密码只在本机子进程内解密，并在进程间管道传递，不出现在 shell 参数、脚本源码、
SSH 命令行、标准输出或远端文件中。不要给包含凭据的变量加调试打印。

Python 工程可用 `importlib.util.spec_from_file_location` 按上述绝对路径加载
`reconstruction_ssh.py`，调用 `connect()` 得到 Paramiko SSHClient，然后使用
`client.exec_command(...)` 或 `client.open_sftp()`。用 context manager 关闭连接。
不要转发本机凭据到登录节点来连接其他服务，也不要假定其他工程的命令已被授权。

### 调度与空间

- 本工程的所有重建任务必须位于 `/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/` 或其子目录。FOV120 使用该工程的 `experiments/FOV120_20260924/` 独立工作区。
- `gpugpu` 已查得单作业最多 8 节点；账号关联 `GrpTRES=gres/gpu=100`，5090 每节点 8 卡。
  这是配置快照，提交前复核 QOS、当前资源和队列，不把登录横幅的通用额度当作实际账号限制。
- FOV120 的 1e9 全网格六路实测采用 4 节点 × 每节点 2 张 5090（8 卡总数）：
  单节点整 8 卡等待时间长时，可申请多个各有 2 张空卡的节点。
  `sbatch -N 4 --gres=gpu:2` 必须与 `FOV120_GPUS_PER_NODE=2` 一致，
  `reconstruct.sh` 会检查分配卡数；1e10 根据实际接受事件和显存重新决定卡数。
  1e9 Contrast 短程峰值 reserved 约 9.97 GiB/卡，不能直接外推为 1e10 显存需求。
- GPU 作业用 sbatch，登录节点用于编辑/编译/传输；不要在登录节点跑重建。
- 已有 torch 环境的作业使用 `module load cuda/12.9`、`module load miniforge3/25.11.0-1`，
  再 source conda.sh、`conda activate torch`。批处理和非登录 SSH shell 应先 `source /etc/profile.d/modules.sh` 再加载模块；
  已有 sbatch 环境可用，不能把交互 shell 的 PATH 直接当作计算节点环境。
- `squeue -u scxi717`、`scontrol show job JOBID` 用于只读检查。不要取消或覆盖其他工程的作业。
- 计算节点不能联网；依赖在登录节点预备。提交前核实空间和配额，`/ssd/scxi717` 本次访问被拒绝，不用于部署。
- 不自动改投高价 hp 队列；FOV120 使用普通 gpu_5090。

## 3. maty：私钥口令与 ssh-agent

原始密钥位于 `C:\Users\Lipeize\Desktop\超算平台\id_rsa_2048`（这是私钥；`.pub` 才是公钥）。
当前使用权限已收紧的**加密副本** `%USERPROFILE%\.ssh\maty_id_rsa_2048`。
原文件未改写，私钥口令从未写入工程。服务器已接受对应公钥。

如 agent 尚未启动，在管理员 PowerShell 执行一次：

```powershell
Set-Service ssh-agent -StartupType Manual
Start-Service ssh-agent
```

随后回到同一 Windows 用户的普通 PowerShell：

```powershell
ssh-add "$env:USERPROFILE\.ssh\maty_id_rsa_2048"
ssh-add -l
ssh -o BatchMode=yes maty@192.168.11.1 hostname
```

ssh-add 提示的是**私钥解锁口令**，不必等同于服务器账号密码。`Identity added`
后其他同用户进程可通过 agent 签名登录。本次仅指定加密私钥路径并启用
`IdentitiesOnly=yes` 时 OpenSSH 无法匹配 agent 身份；直接使用已解锁的 agent 即可。
若 agent 中身份很多，应正确配置对应公钥身份文件后再限制身份，避免多次尝试失败。
不导出私钥明文、不把口令放入参数、不将密钥复制到远端。

项目原目录：
`/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/Geant4Sim`。
当前独立目录：同级 `FOV120_20260924`；Geant4 源码在其 `Geant4Sim/Geant4Code/`。

集群环境：

```bash
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
```

CMake 使用 `/apps/tools/cmake/v3.25.2/bin/cmake`，显式指定
`/apps/compilers/gcc/v12.2.0/bin/gcc` 与 `g++`，避免系统 GCC 4.8.5。
Python3 为 3.10.4，numpy 1.24.3；不要依赖 Python 3.11 才提供的 hashlib.file_digest。
使用 sbatch/cnmix；单线程 gamma01 对应一个 CPU，任务隔离工作目录、种子和输出。
当前分区拒绝了 `--mem=4G` 请求，脚本沿用原工程内存默认值，不能照搬其他集群模板。

## 4. 65114：现有 SSH 配置与证书

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=10 65114_lipeize 'hostname; whoami'
```

现有 `%USERPROFILE%\.ssh\config` 中该别名使用 `lipeize` 私钥及 `lipeize-cert.pub`
证书，`IdentitiesOnly yes`。不要覆盖原配置，不复制密钥到工程；证书过期时使用原有
授权流程更新，不改服务器权限或绕过认证。具体文件路径以本机 config 为准。

本实验目录 `/home/lipeize/JSCC_FOV120_20260924`，使用其独立 `.venv`。
启动前检查 `nvidia-smi`，本次仅 GPU 0 空闲并被使用；其余卡上的任务不属于本次操作。
`nohup` 长任务必须重定向日志，保存 PID、命令、输入/二进制哈希与完成标记。
不要按进程名称批量终止任务。跨项目传输优先 SFTP/SCP，核对目的地和文件哈希。

## 5. 安全和接续检查清单

1. 阅读本说明后，先用 hostname/whoami/pwd 验证身份和目录，不以截图/旧日志推断当前连接成功。
2. 每个工程独立实验目录，运行前核对现有文件，禁止覆盖已完成 worker 或改写基线数据。
3. 密码、私钥、加密凭据、令牌、会话导出文件和 `.env` 不进 Git；路径和非秘密连接方法可以写入文档。
4. 同一主机的新工程可以复用当前用户的认证；新电脑/新用户必须重新授权，不能靠复制 DPAPI 文件解密。
5. 密钥撤销使用对应身份的 ssh-add -d，不用 ssh-add -D 清空其他工程身份。
   不再需要密码连接时，可删除本机指定 credential.xml，并按平台流程更换账号密码。
6. 新工程自己记录作业号、节点、环境、源代码 commit、数据哈希与失败状态。状态快照不是永久完成证明。
7. 推送前运行 `python tools/security/audit_git_payload.py`；该检查是模式和大小筛查，不是无泄漏的数学保证。
