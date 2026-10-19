"""Rebuild project navigation from local files only; no scientific execution."""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/ac225'
H = 'experiments/ELLIPSE500x300_H120'
R = H+'/reports/NEMA_Body_H60/'
D = 'derenzo_equal_regions_mono_5e9_20261016'
P = 'Analysis_Plots/research/'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def entry(identity, title, topic, status, kind, entries, evidence=(), local_data=(), figures=(), **extra):
    return dict(id=identity, title=title, topic=topic, status=status, data_kind=kind,
                entries=list(entries), evidence=list(evidence), local_data=list(local_data), figures=list(figures), **extra)


def experiments():
    items = [
        entry('nema-jscc-5e10', 'NEMA JSCC 5e10：完整验证通过，正式时长阻断', 'nema ac225 jscc 218 440',
              'formal-held-time', 'actual-Geant4-dual-energy-proxy',
              [R+'jscc_geant4_5e10_10000/README.md', R+'jscc_geant4_5e10_10000/CONTRACT.md', R+'jscc_geant4_5e10_10000/RUNBOOK.md'],
              [R+'jscc_geant4_5e10_10000/validation_acceptance.json', R+'jscc_geant4_5e10_10000/production_5090_8x4_queue_measured_time_hold_20261019.json'],
              [H+'/generated/jscc_geant4_5e10_10000'],
              jobs=dict(validation=1704559, verification=1704566, formal=1704572),
              channels=['440_SinglePhoton', '218_SinglePhoton_CrossTalkCorrected', '440_ComptonOnly'],
              iterations=10000, save_step=50, complete_formal=False),
        entry('nema-jscc-5e9-v5', 'NEMA JSCC 5e9 连续核v5：历史10000次正式基准', 'nema ac225 jscc 218 440',
              'delivered-historical', 'actual-Geant4-dual-energy-proxy',
              [R+'compton_energy_probability_v5_5e9_full10000/ACCEPTANCE.md', 'docs/DUAL_ENERGY_BASELINE.md'],
              local_data=[H+'/generated/compton_energy_probability_v5_5e9_full10000/formal_results/1669255/continuous_energy'],
              jobs=dict(formal=1669255), iterations=10000, save_step=50,
              note='历史六路含已退休跨能量求和；后续仅采用获授权的分能量通道和当前ROI。'),
    ]
    for dose, actual_job, model_job in [('5e9', 1679415, 1680255), ('5e10', 1681346, 1683357)]:
        for mode, job in [('g4', actual_job), ('poisson', model_job)]:
            name = f'ehe_spect_{dose}_200' if mode == 'g4' else f'ehe_forward_poisson_{dose}_200'
            doc = R+name+('/RESULTS.md' if (ROOT/(R+name+'/RESULTS.md')).exists() else '/README.md')
            items.append(entry(f'nema-ehe-{mode}-{dose}', f'NEMA EHE {dose}：'+('实际Geant4' if mode=='g4' else '矩阵前投影+Poisson参考'),
                'nema ac225 ehe 218 440 '+mode, 'delivered' if mode=='g4' else 'reference-only',
                'actual-Geant4-dual-energy-proxy' if mode=='g4' else 'matrix-forward-plus-Poisson',
                [doc], [R+name+'/formal_summary.json'], [H+'/generated/'+name+'/results/formal'],
                jobs=dict(formal=job), iterations=200,
                channels=['440_SinglePhoton', '218_SinglePhoton_CrossTalkCorrected'],
                note='原物理响应HOLD保留；执行验收不代表物理偏差消失。历史求和只读，不进入新比较。'))
    for dataset in ['JSCC_218', 'JSCC_440', 'EHE_218', 'EHE_440']:
        system, energy = dataset.split('_')
        base = H+'/reports/Derenzo/'+D+'/'+dataset+'/'
        proof = base+'formal_acceptance.json'
        done = ((ROOT/proof).exists() and read(proof).get('passed') is True
                and read(proof).get('strict_fetch_passed') is True)
        registered = (ROOT/(base+'formal_job.json')).exists()
        jobs = {}
        for stage in ['selection', 'selection_verify', 'validation', 'validation_verify', 'formal', 'formal_verify']:
            path = base+stage+'_job.json'
            if (ROOT/path).exists():
                jobs[stage] = read(path)['job']
        evidence = [base+('formal_acceptance.json' if done else 'formal_job.json' if registered else 'transport_acceptance.json')]
        channels = [energy+'_SinglePhoton']
        if dataset == 'JSCC_440':
            channels.append('440_SinglePlusCompton')
        figures = [P+'derenzo_completed_5e9_20261019/figures/final_mip.png'] if done else []
        items.append(entry('derenzo-'+dataset.lower().replace('_', '-'), f'Derenzo {system} {energy} keV：独立单能5e9',
            'derenzo monoenergy '+system.lower()+' '+energy, 'delivered' if done else 'formal-registered',
            'actual-Geant4-monoenergy', [H+'/reports/Derenzo/'+D+'/README.md'], evidence,
            [H+'/generated/'+D+'/'+dataset], figures, jobs=jobs, channels=channels,
            iterations=10000 if system=='JSCC' else 200, save_step=50 if system=='JSCC' else 1,
            note='每个探头/能量独立5e9 gamma；两个源不是同一次225Ac衰变链，218背景为零。'))
    items += [
        entry('derenzo-design-selected', 'Derenzo等面积区域：实际输运采用的唯一设计', 'derenzo phantom geometry 218 440',
              'design-selected', 'authoritative-analytic-geometry',
              [P+'derenzo_218_440_design_20261015_equal_regions/design_notes_zh.txt'],
              [P+'derenzo_218_440_design_20261015_equal_regions/data/derenzo_218keV_geometry.json', P+'derenzo_218_440_design_20261015_equal_regions/data/derenzo_440keV_geometry.json'],
              [H+'/generated/'+D+'/macros'], [P+'derenzo_218_440_design_20261015_equal_regions/figures/derenzo_440_218_design.png'],
              note='218：6/8/10/12/14/16mm，共114棒；440：10/12/14/16/18/20mm，共53棒。'),
        entry('derenzo-completed-figures', 'Derenzo已完成三组的真值、MIP、切片和迭代图', 'derenzo visualization figures 218 440',
              'analysis-complete', 'derived-from-strictly-delivered-results',
              [P+'derenzo_completed_5e9_20261019/README.txt', 'tools/research/plot_derenzo_completed.py'],
              [P+'derenzo_completed_5e9_20261019/data/scientific_qa.json', P+'derenzo_completed_5e9_20261019/data/visual_qa.json'],
              figures=[P+'derenzo_completed_5e9_20261019/figures/final_mip.png', P+'derenzo_completed_5e9_20261019/figures/iteration_mip_gallery.png']),
        entry('nema-current-roi', 'NEMA当前1.5mm内缩球中心ROI及统一完整背景', 'nema roi analysis', 'analysis-complete',
              'current-evaluation-policy', [H+'/reports/nema_interior_roi_20261010/README.md', H+'/nema_roi_policy.py', H+'/reconstruction_output_policy.py'],
              [H+'/reports/nema_interior_roi_20261010/scientific_acceptance.json']),
        entry('nema-source-truth', 'NEMA H60球体、肺、218/440空间分布', 'nema truth phantom geometry 218 440',
              'authoritative-source', 'analytic-3D-source', [R+'README.md'], [R+'manifest.json'],
              [H+'/generated/NEMA_Body_H60/truth_3mm.npz'], [P+'nema_structure_20261014/figures/nema_structure_218_440.png']),
        entry('compton-v5-legacy', '440连续材料能量核：legacy独立校准及2000次配对', 'calibration compton 440 v5',
              'accepted-reference', 'independent-calibration', [R+'compton_energy_probability_v5_5e9/CALIBRATION_ACCEPTANCE.md', R+'compton_energy_probability_v5_5e9/ACCEPTANCE.md']),
        entry('compton-v5-ideal', 'ideal 1e9连续核对照；与legacy剂量实验分开', 'calibration compton 440 v5 ideal',
              'historical-reference', 'paired-physics-study', [R+'compton_energy_probability_v5/ACCEPTANCE.md']),
        entry('first-scatter-v2', '首散射事件定义配对研究', 'compton diagnostics first scatter',
              'historical-reference', 'paired-event-definition-study', [R+'compton_first_scatter_v2/ACCEPTANCE.md']),
        entry('event-physics-audit', '全局事件算子、测度及物理响应审计', 'physics audit numerical stability',
              'accepted-diagnostic', 'physics-and-numerical-audit', [R+'process_list_global_audit_v4/REPORT.md']),
        entry('spike-ablation', '边界绑定、Huber和尖峰历史消融', 'nema regularization ablation',
              'stopped-historical', 'historical-ablation', [R+'spike_ablation/README.md']),
        entry('response-cut3-old', '旧response mismatch门控：保留历史身份', 'compton q gate response mismatch',
              'retired-reference', 'historical-selection', [R+'response_mismatch_cut3_v1/README.md']),
        entry('fine-response-v3', '已停止精细响应场与清理证明', 'compton fine response geometry',
              'stopped-not-complete', 'incomplete-physics-study', [R+'compton_response_geometry_v3/STOP_AND_CLEANUP_20261005.md']),
        entry('advisor-progress', '成像场景、JSCC/EHE设计及既有NEMA结果说明图', 'figures advisor nema design',
              'presentation-reference', 'derived-presentation', [P+'advisor_progress_20261014/使用说明.txt'],
              figures=[P+'advisor_progress_20261014/figures/01_imaging_scene.png', P+'advisor_progress_20261014/figures/02_detector_design.png', P+'advisor_progress_20261014/figures/03_nema_mip_iterations.png']),
    ]
    covered = {Path(p).parent.as_posix() for item in items for p in item['entries'] if Path(p).suffix}
    for folder in sorted((ROOT/R).iterdir()):
        relative = folder.relative_to(ROOT).as_posix()
        if not folder.is_dir() or relative in covered:
            continue
        candidates = [folder/n for n in ['README.md','ACCEPTANCE.md','REPORT.md','PRODUCTION.md']]
        document = next((p for p in candidates if p.is_file()), folder)
        items.append(entry('history-'+folder.name.lower().replace('_','-'), folder.name+'：历史研究/诊断目录',
              'nema historical '+folder.name, 'historical-see-evidence', 'historical-study-or-diagnostic',
              [document.relative_to(ROOT).as_posix()],
              note='按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。'))
    for version in ['derenzo_218_440_design_20261014', 'derenzo_218_440_design_20261015', 'derenzo_218_440_design_20261015_sparse36']:
        items.append(entry('history-'+version.replace('_','-'), version+'：未采用的历史设计',
              'derenzo design historical geometry', 'design-not-selected', 'historical-geometry-design',
              [P+version+'/README.md'], note='不是本轮输运真值；无对应新模拟完成声明。'))
    return items


def component_catalog():
    candidates = [
        ('experiment-ellipse', '当前500×300×120mm三维实验', [H+'/README.md']),
        ('simulation-jscc', 'JSCC Geant4输运', ['Geant4Sim/Geant4Code/README.md']),
        ('simulation-ehe', 'EHE Geant4输运', ['Geant4Sim/Geant4Code_EHE/README.md']),
        ('source-measure', '极坐标体积测度与密度单位', ['GenProj/POLAR_SOURCE_MEASURE.md', 'GenProj/README.md']),
        ('reproduction', '历史通用复现入口', ['Reproduction/README.md']),
        ('local-compatibility', '根目录历史本地重建接口', ['main_local_multi_energy_cntstat.py']),
        ('legacy-results', '旧60mm本地结果目录说明', ['Results/README.md']),
        ('research-strategy', '与平面信息响应/系统优化工程的桥接', ['docs/RESEARCH_STRATEGY.md']),
        ('remote-access', '超算连接规范；凭据位于工程外', ['docs/REMOTE_COMPUTE_ACCESS.md']),
        ('matrix-engine', 'CUDA系统矩阵与参数引擎', ['Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/ReadMe.md']),
        ('factor-conversion', '密度基底因子转换与标定', ['Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/GenFactors/README.md']),
        ('compton-sensitivity', 'Compton密度基底灵敏度', ['Auxiliary_Studies/Sensitivity_SPECT_PolarCoor/README.md']),
        ('event-order-study', '独立事件顺序研究', ['Auxiliary_Studies/EventOrderInference_Experiment/README.md']),
        ('historical-fov120', '另一代FOV120几何实验', ['experiments/FOV120/README.md']),
        ('distributed-interfaces', '通用历史分布式接口', ['distributed/dual_energy_compton_slurm']),
    ]
    return [dict(id=i, title=t, entries=p) for i,t,p in candidates]


def group(path):
    name = path.name.lower()
    text = path.as_posix()
    if '/reports/' in text or '/scripts/' in text and text.startswith('Analysis_Plots/'):
        return 'frozen-or-artifact-copy'
    if name.startswith('test_') or name.startswith('verify') or name.endswith('_verify.py'):
        return 'verification-tests'
    if any(t in name for t in ['workflow', 'queue', 'resume_', 'monitor_', 'launch', 'submit']):
        return 'execution-control'
    if any(t in name for t in ['plot_', 'analy', 'compare_', 'report', 'visual', 'summarize']):
        return 'analysis-figures'
    if any(t in name for t in ['transport', 'collect', 'detector', 'eventaction', 'runaction', 'steppingaction']):
        return 'transport'
    if any(t in name for t in ['phantom', 'geometry', 'macro', 'fov_', 'source', 'config']):
        return 'geometry-source'
    if any(t in name for t in ['calibrat', 'response', 'probability', 'measure', 'kernel', 'sensi', 'diagnos']):
        return 'calibration-physics'
    if any(t in name for t in ['mlem', 'recon', 'osem', 'run_', 'main_', 'operator']):
        return 'reconstruction'
    return 'shared-or-legacy-source'


def sources():
    files = set()
    for args in [['git', 'ls-files', '-z'], ['git', 'ls-files', '--others', '--exclude-standard', '-z']]:
        files.update(Path(p) for p in subprocess.check_output(args, cwd=ROOT).decode().split('\0') if p)
    code = {'.py', '.cu', '.cc', '.cpp', '.h', '.hpp', '.m', '.ps1', '.sh', '.mac'}
    result = []
    for path in sorted(files):
        if path.suffix.lower() not in code or not (ROOT/path).is_file():
            continue
        purpose = ''
        if path.suffix == '.py':
            try:
                doc = ast.get_docstring(ast.parse((ROOT/path).read_text(encoding='utf-8-sig')))
                purpose = ' '.join((doc or '').split())[:120]
            except (SyntaxError, UnicodeError):
                purpose = '历史源码；按原解释器和依赖阅读'
        result.append(dict(path=path.as_posix(), group=group(path), purpose=purpose or path.stem))
    return result


def directories():
    rows = []
    roots = [ROOT, ROOT/'experiments', ROOT/H, ROOT/H/'reports', ROOT/R,
             ROOT/H/'reports/Derenzo', ROOT/H/'generated', ROOT/'Analysis_Plots/research']
    for parent in roots:
        if not parent.is_dir():
            continue
        for path in sorted(parent.iterdir()):
            if not path.is_dir() or path.name.startswith(('.', '__')):
                continue
            name = path.relative_to(ROOT).as_posix()
            local = ('/generated' in name or name in ['Factors','CntStat','List','tmp'])
            rows.append(dict(path=name, local_only=local,
                             role='local-runtime-data' if local else 'source-report-or-artifact-directory'))
    return sorted({row['path']:row for row in rows}.values(), key=lambda r:r['path'])


def link(target, document):
    return os.path.relpath(ROOT/target, document.parent).replace('\\', '/')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    items, components, code, folders = experiments(), component_catalog(), sources(), directories()
    catalog = dict(schema_version=1, snapshot_at=now, snapshot_semantics='local evidence only; no automatic job action',
                   convention='Ac225 label denotes separately documented218/440gamma proxy fields, not full decay-chain events',
                   cross_energy_image_sums_currently_authorized=False,
                   experiments=items, components=components)
    write(OUT/'catalog.json', catalog)
    write(OUT/'source_index.json', dict(schema_version=1, snapshot_at=now,
          note='Inventory is navigation, not authorization to run historical scripts.',
          groups=dict(Counter(item['group'] for item in code)), files=code))
    write(OUT/'directory_index.json', dict(schema_version=1, snapshot_at=now, directories=folders,
          note='Shallow directory inventory only; ignored runtime arrays are not scanned or moved.'))
    doc = OUT/'EXPERIMENTS.md'
    lines = ['# 实验与证据索引', '', f'本地证据快照：{now}。状态不是实时Slurm查询；注册作业不等于已经开始计算。', '',
             'Git保存源码、小型登记、验收和查询入口；`generated/`及图片属于本地数据，克隆后需按记录恢复。', '',
             '| 查询ID | 实验 | 状态 | 报告入口 |', '|---|---|---|---|']
    for item in items:
        p = item['entries'][0]
        lines.append(f"| `{item['id']}` | {item['title']} | {item['status']} | [打开]({link(p,doc)}) |")
    lines += ['', '```powershell', 'python tools/project_index.py list --topic nema',
              'python tools/project_index.py show nema-jscc-5e10',
              'python tools/project_index.py show derenzo-jscc-440',
              'python tools/project_index.py find compton', '```', '',
              '状态、数据目录、图像、通道及作业号的完整字段见[catalog.json](catalog.json)。',
              '`delivered`=正式独立验收且严格取回；`formal-registered`=仅登记后续请求；`formal-held-time`=受时限门槛暂挂；`reference-only`=独立模型参考；`stopped-*`/`historical-*`不自动恢复。',
              'Poisson组只作为独立模型参考；默认实际成像展示使用Geant4组。历史跨能量求和、旧ROI和停止研究均不自动恢复。', '']
    doc.write_text('\n'.join(lines), encoding='utf-8')
    counts = Counter(item['group'] for item in code)
    text = ['# 代码分类与查询', '', f'索引包含{len(code)}个现存源码/宏/脚本文件。完整路径及模块说明见[source_index.json](source_index.json)。', '',
            '| 分类 | 数量 |', '|---|---:|']+[f'| `{k}` | {v} |' for k,v in sorted(counts.items())]
    text += ['', '```powershell', 'python tools/project_index.py files --group execution-control --topic derenzo',
             'python tools/project_index.py files --group verification-tests --topic 5e10',
             'python tools/project_index.py find single_mlem', '```', '',
             '分类是查找辅助；`frozen-or-artifact-copy`是冻结/绘图副本，不是新的执行入口。查询不导入、调用或提交这些脚本。', '',
             '## 常用当前入口', '', '| 用途 | 文件 |', '|---|---|']
    selected = [H+'/derenzo_campaign_workflow.py', H+'/derenzo_campaign_transport.py', H+'/derenzo_campaign_reconstruction.py',
                H+'/derenzo_campaign_verify.py', H+'/resume_reconstruction_controllers_20261019.py', H+'/jscc_5e10_5090_production_queue.py',
                H+'/jscc_5e10_formal_queue_gate.py', H+'/single_checkpoint_mlem.py', H+'/torch_active_operator.py',
                H+'/nema_roi_policy.py', H+'/reconstruction_output_policy.py', 'tools/research/plot_derenzo_completed.py',
                'tools/research/analyze_nema_interior_roi.py']
    source_map = {row['path']:row for row in code}
    for path in selected:
        row = source_map.get(path)
        if row:
            text.append(f"| {row['group']} | [{Path(path).name}]({link(path,OUT/'SOURCE_INDEX.md')}) |")
    (OUT/'SOURCE_INDEX.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    snapshot = dict(snapshot_at=now, semantics='Local acceptance/registration snapshot, not live Slurm', studies=[])
    for item in items:
        if item['id'].startswith(('nema-jscc', 'nema-ehe', 'derenzo-jscc', 'derenzo-ehe')):
            proofs = {}
            for p in item['evidence']:
                if (ROOT/p).is_file():
                    proofs[p] = hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            snapshot['studies'].append(dict(id=item['id'], status=item['status'],
                 jobs=item.get('jobs', {}), evidence_sha256=proofs))
    write(OUT/'status_snapshot.json', snapshot)
    status = ['# 当前状态与阻点', '', f'快照时间：{now}，Asia/Shanghai。来自本机已登记/验收证据，不是实时Slurm查询。', '',
              '| 研究 | 当前阶段 | 主要作业 |', '|---|---|---|']
    for item in snapshot['studies']:
        status.append(f"| `{item['id']}` | {item['status']} | "+', '.join(f'{k}={v}' for k,v in item['jobs'].items())+' |')
    hold = R+'jscc_geant4_5e10_10000/production_5090_8x4_queue_measured_time_hold_20261019.json'
    measured = read(hold)
    status += ['', '## NEMA JSCC5e10', '',
       '完整32GPU validation10任务1704559和独立验收1704566已成功退出，结果严格取回并生成本次authority。正式1704572仍暂挂、没有10000次正式结果。', '',
       f"实测吞吐外推三路10000次共{measured['estimated_10000_total_hours']:.2f}小时（准备只计一次），1.5倍余量需{measured['required_margin_walltime_minutes']}分钟。现有启动脚本的48小时总/24小时单阶段门槛不足，恢复控制器按门槛退出；这不是超算平台最大时限的声明。不得绕过门槛或把本次10次验证图当作10000次正式图。", '',
       f'[原始时长证据]({link(hold,OUT/"CURRENT_STATUS.md")})；后续需独立执行时限或分段方案修复，保留原请求与成功输出。', '',
       '## Derenzo', '',
       'JSCC218正式10000次、EHE218/440正式各200次已验收并严格取回。JSCC440筛选已验收取回，完整验证1713939、验收1713940和暂挂正式1713943已登记；尚无formal_acceptance证明其正式完成。', '',
       '此前只读SSH通道超时和过期PID身份问题已用独立执行控制恢复处理，Derenzo一次性流程继续，定时自动任务未恢复。活跃PID需按实际进程身份核对，不把旧登记状态当作进程存活证明。', '',
       '## 已完成历史组与分析边界', '',
       'NEMA JSCC5e9 v5及EHE实际5e9/5e10已完成；EHE前投影+Poisson是独立模型参考。历史六路/三路求和只读，当前只分析获授权分能量通道。图像噪声、小球、EHE物理响应HOLD及原生网格限制均保留。', '',
       '新Derenzo图形已通过输入SHA、形状、迭代帧、真值体积和独立视觉QA；EHE明显边缘伪影如实保留，不等同分辨率通过。', '',
       '具体数据、图形和验收入口见[实验表](EXPERIMENTS.md)。机器快照及证据SHA见[status_snapshot.json](status_snapshot.json)。', '']
    (OUT/'CURRENT_STATUS.md').write_text('\n'.join(status), encoding='utf-8')
    print(json.dumps(dict(experiments=len(items), components=len(components), sources=len(code), directories=len(folders), snapshot_at=now)))


if __name__ == '__main__':
    main()
