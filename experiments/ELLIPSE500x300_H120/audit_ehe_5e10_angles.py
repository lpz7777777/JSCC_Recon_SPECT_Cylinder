"""Read-only angular-source identity audit of the registered 5e10 acquisition."""
import hashlib, time
from pathlib import Path
from ehe_common import digest, read, write
from ehe_5e10_workflow import DATA, REPORT, CPU_BASE, connection, command


def audit():
    freeze = read(REPORT/'transport_freeze.json')
    config = read(DATA/freeze['payload_dir']/'config.json')
    registry = read(DATA/'simulation/jobs.json')
    original = read(REPORT.parent/'ehe_spect_5e9_200/transport_repair_freeze.json')
    original_payload = DATA.parent/'ehe_spect_5e9_200'/original['payload_dir']
    source_name = 'Geant4Code_EHE/src/PrimaryGeneratorAction.cc'
    source = original_payload/source_name
    expected_source = config['original_transport_source_sha256'][source_name]
    assert digest(source) == expected_source
    source_text = source.read_text()
    anchors = [
        'fXcatGun = new G4ParticleGun(1)',
        'const auto cosTheta = 2 * G4UniformRand() - 1;',
        'const auto phi = twopi * G4UniformRand();',
        'fXcatGun->SetParticleMomentumDirection(G4ThreeVector(',
        'sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta)',
    ]
    assert all(anchor in source_text for anchor in anchors)
    macros = []
    with connection('maty') as client:
        job = read(REPORT/'transport_job.json')['job']
        queue = command(client, 'squeue -h -j '+str(job)+' -o "%i|%T|%M|%R"')
        account = command(client, 'sacct -j '+str(job)+' --parsable2 --noheader --format=JobID,State,ExitCode,Elapsed,AllocTRES,NodeList')
        with client.open_sftp() as sftp:
            def remote_bytes(path):
                with sftp.open(path, 'rb') as stream:
                    return stream.read()
            actual_source = remote_bytes(config['original_transport_root']+'/'+source_name)
            assert hashlib.sha256(actual_source).hexdigest() == expected_source
            # The actual executable was already fully accepted; this binds its
            # current bytes to that same accepted build, without a new simulation.
            binary_sha = command(client, 'sha256sum '+repr(config['binary_path'])).split()[0]
            assert binary_sha == config['binary_sha256']
            for entry in registry['macros']:
                path = DATA/'simulation'/entry['path']
                raw = path.read_bytes()
                remote = remote_bytes(CPU_BASE+'/simulation/'+entry['path'])
                assert raw == remote and digest(path) == entry['sha256']
                lines = [line.strip() for line in raw.decode('ascii').splitlines()
                         if line.strip() and not line.lstrip().startswith('#')]
                commands = sorted(set(line.split()[0] for line in lines))
                assert commands == ['/run/beamOn','/xcat/add','/xcat/angle','/xcat/centerY','/xcat/clear']
                assert lines.count('/run/beamOn 50000000') == 1
                assert sum(line.startswith('/xcat/add ') for line in lines) == 3937
                angle = [line for line in lines if line.startswith('/xcat/angle ')][0]
                assert float(angle.split()[1]) == (entry['view']-1)*18
                macros.append(dict(view=entry['view'], sha256=entry['sha256'],
                    commands=commands, source_position_rotation_degrees=(entry['view']-1)*18,
                    explicit_direction_range_commands=[], xcat_source_boxes=3937,
                    beam_on=50000000))
            actual_worker = None
            try:
                raw = remote_bytes(CPU_BASE+'/transport/worker_0000/source.mac')
            except FileNotFoundError:
                pass
            else:
                registered = (DATA/'simulation'/registry['jobs'][0]['macro']).read_bytes()
                assert raw == registered.replace(b'\r\n',b'\n')
                actual_worker = dict(index=0, sha256=hashlib.sha256(raw).hexdigest(),
                    registered_macro_sha256=registry['jobs'][0]['macro_sha256'],
                    only_crlf_to_lf=True)
    proof = dict(passed=True, audited_epoch=time.time(), job=job,
        actual_binary_sha256=binary_sha, primary_generator_sha256=expected_source,
        source_registry_sha256=freeze['source_registry_sha256'], macros=macros,
        direction_branch='fUseXcat: G4ParticleGun(1), not GPS angular distribution',
        emission_solid_angle_sr='4*pi', isotropic=True,
        cosine_distribution='uniform [-1,1]', azimuth_distribution='uniform [0,2*pi)',
        source_position_rotation_is_emission_restriction=False,
        dose_equivalent_multiplier=1, registered_actual_primary_photons=50000000000,
        full_sphere_equivalent_primary_photons=50000000000,
        actual_started_worker_macro=actual_worker, squeue=queue, sacct=account,
        method='All 20 local/deployed macro bytes plus frozen/current generator and accepted executable SHA; static direction law, no invented runtime angular samples',
        audit_code_sha256=digest(Path(__file__)))
    write(REPORT/'source_angular_audit.json', proof)
    print('ALL20_MACROS_FULL_4PI_DOSE_MULTIPLIER_1', flush=True)
    print(queue, account, sep='\n', flush=True)


if __name__ == '__main__':
    audit()
