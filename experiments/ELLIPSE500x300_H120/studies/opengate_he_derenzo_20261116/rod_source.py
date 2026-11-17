"""Analytic equal-density, vacuum Derenzo sources; no source voxelization."""
from pathlib import Path
import json
import numpy as np
from opengate_ehe_common_20261113 import read, save, sha, now


def add_sources(sim, geometry, job):
    import opengate as gate
    from scipy.spatial.transform import Rotation
    rods = geometry['rods']
    counts = job['rod_primary_counts']
    if len(rods) != len(counts) or sum(counts) != job['primaries']:
        raise ValueError('Rod allocation does not close to exact imaging budget')
    frame = sim.add_volume('Box', 'source_frame')
    frame.size = [300, 300, 120]
    frame.material = 'G4_Galactic'
    frame.rotation = Rotation.from_euler('z', job['angle_deg'], degrees=True).as_matrix()
    for rod, count in zip(rods, counts):
        if count == 0:
            continue
        src = sim.add_source('GenericSource', 'rod_' + str(rod['rod_id']))
        src.attached_to = frame.name
        src.particle = 'gamma'
        src.number_of_primaries = int(count)
        src.energy.type = 'mono'
        src.energy.mono = job['energy_keV'] * gate.g4_units.keV
        src.position.type = 'cylinder'
        src.position.radius = rod['diameter_mm'] / 2
        src.position.dz = (rod['z_max_mm'] - rod['z_min_mm']) / 2
        src.position.translation = [rod['x_mm'], rod['y_mm'], rod['z_mm']]
        src.direction.type = 'iso'
    return frame


def source_check(inputs, output):
    """6,144 tagged numerical histories INSIDE worker 0, excluded from acquisition."""
    import opengate as gate
    import uproot
    from scipy.spatial.transform import Rotation
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    geometry = read(Path(inputs) / 'geometry.json')
    rod = geometry['rods'][0]
    records = []
    for angle in (0, 18, 90):
        sim = gate.Simulation()
        sim.output_dir = output / str(angle)
        sim.number_of_threads = 1
        sim.random_seed = 620900001 + geometry['energy_keV'] + angle
        sim.world.size = [2000] * 3
        sim.world.material = 'G4_Galactic'
        sim.physics_manager.physics_list_name = 'G4EmStandardPhysics_option4'
        frame = add_sources(sim, {'rods': [rod]}, dict(rod_primary_counts=[2048],
            primaries=2048, angle_deg=angle, energy_keV=geometry['energy_keV']))
        actor = sim.add_actor('PhaseSpaceActor', 'vertices')
        actor.attached_to = frame.name
        actor.attributes = ['EventID', 'EventPosition', 'ParentID']
        actor.steps_to_store = 'first'
        actor.output_filename = 'vertices.root'
        sim.run(start_new_process=True)
        p = sim.output_dir / 'vertices.root'
        with uproot.open(p) as f:
            a = f['vertices'].arrays(library='np')
        if len(a['EventID']) != 2048 or len(np.unique(a['EventID'])) != 2048 or np.any(a['ParentID'] != 0):
            raise ValueError('Actual primary vertex count/identity mismatch')
        points = np.column_stack([a['EventPosition_' + k] for k in 'XYZ'])
        rot = Rotation.from_euler('z', angle, degrees=True).as_matrix()
        local = points @ rot - [rod['x_mm'], rod['y_mm'], rod['z_mm']]
        rr = (local[:, :2] ** 2).sum(1)
        radius = rod['diameter_mm'] / 2
        if np.max(rr) > radius ** 2 + 1e-7 or np.max(abs(local[:, 2])) > 30 + 1e-7:
            raise ValueError('Actual source outside rotated analytic rod')
        # Broad predeclared 8-sigma-like checks of uniform area and uniform z.
        if abs(rr.mean() / (radius ** 2 / 2) - 1) > .10 or abs((local[:, 2] ** 2).mean() / 300 - 1) > .12:
            raise ValueError('Unexpected cylinder sampling moments')
        if np.linalg.norm(local[:, :2].mean(0)) > radius * .1 or abs(local[:, 2].mean()) > 2:
            raise ValueError('Actual source centroid mismatch')
        records.append(dict(angle_deg=angle, events=2048, radius_mm=radius,
            radial_second_moment_mm2=float(rr.mean()), z_second_moment_mm2=float((local[:, 2] ** 2).mean()),
            vertex_sha256=sha(p)))
    save(output / 'complete.json', dict(passed=True, records=records, created_utc=now(),
        actual_primary_vertex_rotation_verified=True, formal_imaging_primaries=0,
        role='inline source geometry numerical check, never included in imaging counts'))
