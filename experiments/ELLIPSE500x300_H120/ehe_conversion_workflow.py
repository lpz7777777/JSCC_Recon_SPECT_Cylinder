"""Explicitly authorized, separately frozen response conversion recovery."""
import hashlib
import json
import math
from pathlib import Path
import shutil

from ehe_common import DATA, HERE, REPORT, digest, read, verify_files, write


def repair_binding():
    from ehe_5e9_workflow import base
    import re
    path = REPORT / 'response_conversion_freeze.json'
    if not path.exists():
        return None
    value = read(path)
    stop = read(REPORT / 'response_stop_acceptance.json')
    if not stop['passed'] or not stop['explicit_human_stop'] or not stop['fully_exited'] or stop['job'] != 1672966:
        raise ValueError('Source response must have an evidenced human-directed full stop')
    verify_files(REPORT, stop['files'])
    if digest(REPORT / 'response_stop_acceptance.json') != value['source_stop_sha256']:
        raise ValueError('Frozen stop evidence changed')
    if value['producer_release_key'] != '74e129c4460163c5':
        raise ValueError('Conversion repair scientific release identity differs')
    key = value['repair_key']
    if re.fullmatch('[0-9a-f]{16}', key) is None:
        raise ValueError('Invalid immutable repair key')
    expected = dict(source_release=base('gpu') + '/releases/74e129c4460163c5',
                    source_responses=base('gpu') + '/responses',
                    repair_root=base('gpu') + '/conversion_releases/' + key,
                    probe_output=base('gpu') + '/conversion_probe_' + key,
                    response_output=base('gpu') + '/responses_conversion_' + key)
    if any(value[k] != v for k, v in expected.items()):
        raise ValueError('Repair paths must stay within this experiment and preserve original output')
    return value


def setup():
    from ehe_5e9_workflow import base, connection, command, frozen, put_tree, q, GPU_PYTHON
    existing = repair_binding()
    if existing is not None:
        return existing
    stop = read(REPORT / 'response_stop_acceptance.json')
    if not stop['passed'] or not stop['explicit_human_stop'] or not stop['fully_exited'] or stop['job'] != 1672966:
        raise ValueError('Authorized original job must fully exit before repair')
    verify_files(REPORT, stop['files'])
    scientific = frozen('gpu')
    if scientific['release_key'] != '74e129c4460163c5':
        raise ValueError('Only original response conversion may be repaired')
    sources = {n: digest(HERE / n) for n in ('ehe_conversion_io.py', 'ehe_common.py')}
    sources['source_stop_acceptance.json'] = digest(REPORT / 'response_stop_acceptance.json')
    identity = dict(files=sources, producer_release_key=scientific['release_key'],
                    original_pipeline_sha256=scientific['sha256']['ehe_gpu_pipeline.py'],
                    purpose='Conversion I/O only; reuse all computed slabs and complete A218/A440')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    payload = DATA / 'conversion_releases' / key
    payload.mkdir(parents=True, exist_ok=True)
    for name in sources:
        source = REPORT / 'response_stop_acceptance.json' if name == 'source_stop_acceptance.json' else HERE / name
        dest = payload / name
        if dest.exists() and digest(dest) != sources[name]:
            raise ValueError('Existing immutable conversion payload changed')
        if not dest.exists():
            shutil.copy2(source, dest)
    manifest = dict(repair_key=key, **identity)
    manifest_path = payload / 'repair_manifest.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError('Existing conversion manifest changed')
    if not manifest_path.exists():
        write(manifest_path, manifest)
    root = base('gpu') + '/conversion_releases/' + key
    value = dict(repair_key=key, producer_release_key=scientific['release_key'],
                 source_stop_sha256=sources['source_stop_acceptance.json'],
                 original_pipeline_sha256=scientific['sha256']['ehe_gpu_pipeline.py'],
                 original_geometry_sha256=scientific['sha256']['whole_geometry.npz'],
                 source_release=base('gpu') + '/releases/' + scientific['release_key'],
                 source_responses=base('gpu') + '/responses', repair_root=root,
                 probe_output=base('gpu') + '/conversion_probe_' + key,
                 response_output=base('gpu') + '/responses_conversion_' + key,
                 payload_dir=str(payload.relative_to(DATA)), files=sources,
                 manifest_sha256=digest(manifest_path),
                 scope='Only Cartesian/Polar storage conversion and original own S; no Geant4/PE/Scatter rerun')
    with connection('gpu') as c:
        # A user-scoped queue read works after Slurm has removed the old job id.
        queue = command(c, 'squeue -h -u scxi717 -o ' + q('%i|%T'))
        if any(l.split('|')[0] == '1672966' for l in queue.splitlines()):
            raise ValueError('Original job still active')
        command(c, 'mkdir -p ' + q(base('gpu') + '/conversion_releases'))
        with c.open_sftp() as s:
            put_tree(s, payload, root)
        code = 'from ehe_common import *;from pathlib import Path;p=Path(' + repr(root) + ');verify_files(p,read(p/"repair_manifest.json")["files"]);print("IMMUTABLE_CONVERSION_RELEASE_SHA_PASS")'
        print(command(c, 'cd ' + q(root) + ' && ' + q(GPU_PYTHON) + ' -c ' + q(code)))
    write(REPORT / 'response_conversion_freeze.json', value)
    return value


def conversion_script(binding, stage):
    from ehe_5e9_workflow import env_gpu, q, GPU_PYTHON
    output = binding['probe_output'] if stage == 'probe' else binding['response_output']
    script = env_gpu() + 'export PYTHONDONTWRITEBYTECODE=1\ncd ' + q(binding['repair_root']) + '\n' + q(GPU_PYTHON)
    script += ' ehe_conversion_io.py ' + stage + ' --release ' + q(binding['source_release'])
    script += ' --source-responses ' + q(binding['source_responses']) + ' --output ' + q(output)
    script += ' --repair-manifest ' + q(binding['repair_root'] + '/repair_manifest.json')
    if stage == 'complete':
        script += ' --probe ' + q(binding['probe_output'])
    return script + '\n'


def register(stage, binding, minutes):
    from ehe_5e9_workflow import submit
    name = 'response_conversion_probe' if stage == 'probe' else 'response_conversion'
    record = submit('gpu', name, conversion_script(binding, stage), minutes)
    record.update(repair_key=binding['repair_key'], output=binding['probe_output'] if stage == 'probe' else binding['response_output'],
                  source_job=1672966, scope='Conversion only; immutable computed responses reused')
    write(REPORT / (name + '_job.json'), record)
    return record


def advance_conversion(c):
    """Return False while either unique registered storage-only stage is active."""
    from ehe_5e9_workflow import completed, get_json, gpu_stage_accounting
    binding = repair_binding()
    if binding is None:
        raise ValueError('Conversion binding absent')
    path = REPORT / 'response_conversion_probe_job.json'
    if not path.exists():
        register('probe', binding, 30)
        return False
    record = read(path)
    if record['repair_key'] != binding['repair_key'] or record['release_key'] != binding['producer_release_key']:
        raise ValueError('Conversion probe registration differs')
    done, _ = completed(c, record['job'])
    if not done:
        return False
    if not (REPORT / 'response_conversion_probe_resource_acceptance.json').exists():
        gpu_stage_accounting(c, 'response_conversion_probe', binding['probe_output'])
    probe = get_json(c, binding['probe_output'] + '/probe_acceptance.json')
    if not probe['passed'] or probe['repair_key'] != binding['repair_key'] or not probe['bitwise_equal_to_full_bin_original_expression'] or probe['relative_l2'] != 0:
        raise ValueError('Actual full slab conversion numerical gate failed')
    if probe['checked_xy_layers'] != 10 or probe['checked_bins'] != 2312 or probe['checked_points'] != 33010:
        raise ValueError('Full real probe dimensions differ')
    write(REPORT / 'response_conversion_probe_acceptance.json', probe)
    reuse = get_json(c, binding['probe_output'] + '/source_reuse_acceptance.json')
    write(REPORT / 'response_compute_reuse_acceptance.json', reuse)
    path = REPORT / 'response_conversion_job.json'
    if not path.exists():
        # Actual full source audit plus measured real 10-layer interpolation and
        # 973 MB atomic sequential publication. Full publication is exactly 4x.
        io = sum(v['elapsed_seconds'] for v in probe['sequential_publication'].values())
        seconds = math.ceil((probe['source_audit_seconds'] + probe['slab_conversion_seconds'] * 3 + io * 4) * 1.8 + 300)
        record = register('complete', binding, max(15, math.ceil(seconds / 60) + 5))
        record['measured_limit_seconds'] = seconds
        write(path, record)
        return False
    record = read(path)
    if record['repair_key'] != binding['repair_key'] or record['release_key'] != binding['producer_release_key']:
        raise ValueError('Conversion registration differs')
    done, _ = completed(c, record['job'])
    if not done:
        return False
    if not (REPORT / 'response_conversion_resource_acceptance.json').exists():
        gpu_stage_accounting(c, 'response_conversion', binding['response_output'])
    proof = get_json(c, binding['response_output'] + '/conversion_acceptance.json')
    if not proof['passed'] or proof['repair_key'] != binding['repair_key'] or proof['total_layers'] != 40 or proof['reused_probe_layers'] != 10 or proof['converted_layers'] != 30 or not proof['scientific_algorithm_unchanged']:
        raise ValueError('Actual full conversion identity failed')
    write(REPORT / 'response_conversion_acceptance.json', proof)
    if not (REPORT / 'response_resource_acceptance.json').exists():
        old = read(REPORT / 'response_stop_acceptance.json')
        new = read(REPORT / 'response_conversion_resource_acceptance.json')
        write(REPORT / 'response_resource_acceptance.json', dict(passed=True,
            computed_source_job=1672966, computed_source_exit='Human-directed CANCELLED after all 12 calculation receipts',
            complete_conversion_job=record['job'], original_source_stop=old,
            complete_conversion_resource=new, source_reuse_acceptance_sha256=digest(REPORT / 'response_compute_reuse_acceptance.json'),
            conversion_acceptance_sha256=digest(REPORT / 'response_conversion_acceptance.json'),
            response_root=binding['response_output'], scientific_algorithm_unchanged=True,
            physical_gate_pass_claimed=False))
    return True


def response_root():
    from ehe_5e9_workflow import base
    binding = repair_binding()
    return binding['response_output'] if binding is not None else base('gpu') + '/responses'
