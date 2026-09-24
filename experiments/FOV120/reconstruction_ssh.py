"""Run an explicit SSH command using the user's Windows-encrypted credential.

Password is decrypted only into process memory; never passed on a command line.
Host keys must already be present in the user's OpenSSH known_hosts file.
"""
import argparse
import json
from pathlib import Path
import subprocess

import paramiko


def connect():
    credential_path=Path.home()/'.ssh/fov120_paracloud.credential.xml'
    if not credential_path.exists():
        raise RuntimeError('Run save_reconstruction_credential.ps1 locally first.')
    script="""
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$credential=Import-Clixml -LiteralPath (Join-Path $env:USERPROFILE '.ssh/fov120_paracloud.credential.xml')
@{username=$credential.UserName;password=$credential.GetNetworkCredential().Password} | ConvertTo-Json -Compress
"""
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],
                          capture_output=True,text=True,encoding='utf-8',check=False)
    if result.returncode:
        raise RuntimeError('Cannot decrypt the local credential under this Windows user.')
    credential=json.loads(result.stdout)
    client=paramiko.SSHClient()
    client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect('ssh.cn-zhongwei-1.paracloud.com',port=22,
            username=credential['username'],password=credential['password'],
            look_for_keys=False,allow_agent=False,timeout=15,auth_timeout=30)
    finally:
        credential.clear()
        del result
    return client


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command',required=True)
    args=parser.parse_args()
    with connect() as client:
        _,stdout,stderr=client.exec_command(args.command,timeout=60)
        print(stdout.read().decode(errors='replace'),end='')
        import sys
        print(stderr.read().decode(errors='replace'),end='',file=sys.stderr)
        raise SystemExit(stdout.channel.recv_exit_status())
