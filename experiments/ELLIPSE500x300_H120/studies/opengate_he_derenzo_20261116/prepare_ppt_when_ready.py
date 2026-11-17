"""Finite postprocessing only; no Slurm/fetch/reconstruction or final publication."""
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback
import psutil
import workflow as w

OUT=w.ROOT/'Analysis_Plots/research/ac225_results_meeting_he_derenzo_20261116'
RUNTIME=Path('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt/dependencies')
NODE=RUNTIME/'node/bin/node.exe'
SKILL=Path('C:/Users/Lipeize/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11011/skills/presentations')


def main():
    proc=psutil.Process();deadline=time.time()+12.5*3600
    bindings={str(p):w.sha(p) for p in [Path(__file__),w.STUDY/'visualize.py',OUT/'build/edit_deck.mjs',
                OUT/'build/verify_package.py',OUT/'build/render_powerpoint.ps1']}
    registration=dict(pid=proc.pid,create_time=proc.create_time(),cmdline=proc.cmdline(),started_utc=w.now(),
                      deadline_epoch=deadline,status='WAITING_FOR_FOUR_STRICT_DELIVERIES',bindings=bindings,
                      scope='postprocessing only; never submit/fetch/reconstruct; preserve user PPT until visual review')
    w.save(w.REPORT/'ppt_postprocess_registration.json',registration)
    try:
        while time.time()<deadline:
            exitfile=w.REPORT/'controller_exit.json'
            if exitfile.exists():
                result=w.read(exitfile)
                if not result.get('passed'):raise RuntimeError('Scientific controller did not complete successfully')
            if all((w.REPORT/r/'strict_delivery.json').exists() for r in w.ROUTES):break
            time.sleep(45)
        else:raise TimeoutError('Finite postprocess deadline')
        for p,h in bindings.items():
            if w.sha(p)!=h:raise RuntimeError('Bound postprocessing source changed')
        env=dict(os.environ,RUNTIME_NODE_MODULES=str(RUNTIME/'node/node_modules'))
        commands=[
            ([sys.executable,'-X','utf8',str(w.STUDY/'visualize.py')],w.ROOT),
            ([str(NODE),'container_tools/mark_artifact_operation_started.mjs','--operation-kind','edit','--expected-output-count','1','--output-format','pptx'],SKILL),
            ([str(NODE),str(OUT/'build/edit_deck.mjs')],w.ROOT),
            ([sys.executable,'-X','utf8',str(OUT/'build/verify_package.py')],w.ROOT),
            (['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(OUT/'build/render_powerpoint.ps1')],w.ROOT)]
        for index,(command,cwd) in enumerate(commands):
            print('POSTPROCESS_STAGE',index,command,flush=True)
            subprocess.run(command,cwd=cwd,env=env,check=True,timeout=1800)
        w.save(w.REPORT/'ppt_postprocess_exit.json',dict(registration,status='RENDERED_AWAITING_VISUAL_REVIEW',
            passed=True,ended_utc=w.now(),PPT_user_file_replaced=False,visual_QA_passed=False,
            candidate=str(OUT/'output/Meeting_LPZ_20261116_225Ac成像结果_HE_Derenzo.pptx'),
            rendered=str(OUT/'rendered_final')))
    except BaseException:
        w.save(w.REPORT/'ppt_postprocess_exit.json',dict(registration,status='FAILED',passed=False,
            ended_utc=w.now(),error=traceback.format_exc(),PPT_user_file_replaced=False));raise


if __name__=='__main__':main()
