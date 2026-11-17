"""GE HEGP hole specification on the existing finite segmented EHE camera.

This is a hybrid camera, not a complete commercial GE Discovery camera.
All lengths are mm. The Pb entrance remains at +Y=298.5; the original NaI
array moves back by 15.5 to remain flush with the longer collimator.
"""
from __future__ import annotations
import math
import numpy as np

ENTRANCE = 298.5
LENGTH = 66.
FLAT_DIAMETER = 4.
SEPTUM = 1.8
PITCH = FLAT_DIAMETER + SEPTUM
CRYSTAL_Y = ENTRANCE + LENGTH + 5.
RADIUS = FLAT_DIAMETER / 2  # G4Polyhedra historical constructor: apothem


def vertices_mm():
    angles = np.arange(6) * math.pi/3
    return RADIUS/math.cos(math.pi/6)*np.c_[np.cos(angles),np.sin(angles)]


def holes_mm():
    dx = math.sqrt(3)*PITCH
    vertices = vertices_mm()
    points=[]
    for offset in (0,.5):
        for i in range(-40,41):
            for j in range(-40,41):
                p=np.array([(i+offset)*dx,(j+offset)*PITCH])
                if np.all(np.abs(p+vertices) <= np.array([165,82.5])+1e-10):
                    points.append(p)
    return np.asarray(sorted(points,key=lambda p:(p[1],p[0])),dtype=np.float64)


def crystals_mm():
    return np.asarray([[4*(ix-34-.5),CRYSTAL_Y,4*(iz-17-.5)]
                       for ix in range(1,69) for iz in range(1,35)])


def numerical_geometry_check():
    from scipy.spatial import cKDTree
    h=holes_mm();vertices=vertices_mm()
    assert len(h)==1787 and len(np.unique(h,axis=0))==1787
    distances,indices=cKDTree(h).query(h,k=7)
    np.testing.assert_allclose(distances[:,1],PITCH,rtol=0,atol=1e-10)
    # All nearest neighbours lie along flat normals: pitch minus flat diameter
    # is exactly the physical Pb septum, rather than the corner spacing.
    normals=np.c_[np.cos(np.arange(6)*math.pi/3+math.pi/6),
                  np.sin(np.arange(6)*math.pi/3+math.pi/6)]
    near=(distances[:,1:]<PITCH+1e-8)
    vectors=(h[indices[:,1:]]-h[:,None,:])[near]/PITCH
    assert np.max(np.min(np.linalg.norm(vectors[:,None,:]-normals,axis=2),axis=1))<1e-10
    np.testing.assert_allclose(cKDTree(h).query(-h,k=1)[0],0,atol=1e-10)
    assert np.all(np.abs(h[:,None,:]+vertices)<=np.array([165,82.5])+1e-10)
    assert CRYSTAL_Y-5 == ENTRANCE+LENGTH
    assert math.hypot(252,150)<ENTRANCE
    return dict(passed=True,hole_count=len(h),flat_diameter_mm=FLAT_DIAMETER,
        hole_length_mm=LENGTH,minimum_septum_mm=SEPTUM,pitch_mm=PITCH,
        collimator_entrance_y_mm=ENTRANCE,collimator_back_y_mm=ENTRANCE+LENGTH,
        crystal_center_y_mm=CRYSTAL_Y,crystal_shift_from_EHE_mm=15.5,
        source_frame_rotation_minimum_clearance_mm=ENTRANCE-math.hypot(252,150),
        FOV_rotation_minimum_clearance_mm=ENTRANCE-250,
        open_area_fraction=len(h)*2*math.sqrt(3)*RADIUS**2/(330*165),
        full_commercial_GE_camera=False)


def build_camera(sim):
    from scipy.spatial.transform import Rotation
    plate=sim.add_volume('Box','lead_plate')
    plate.size=[330,LENGTH,165];plate.translation=[0,ENTRANCE+LENGTH/2,0]
    plate.material='G4_Pb'
    hole=sim.add_volume('Hexagon','hole');hole.mother=plate.name
    hole.height=LENGTH;hole.radius=RADIUS;hole.material='G4_Galactic'
    hole.rotation=Rotation.from_euler('x',90,degrees=True).as_matrix()
    hole.translation=[[float(x),0.,float(z)] for x,z in holes_mm()]
    crystal=sim.add_volume('Box','crystal');crystal.size=[4,10,4]
    crystal.material='G4_SODIUM_IODIDE';crystal.translation=crystals_mm().tolist()
    return crystal


def actual_solid_check(sim,output):
    """Zero histories: numeric G4-solid audit inside an actual formal worker."""
    import opengate_core as g4
    from opengate_ehe_common_20261113 import save,sha
    from pathlib import Path
    numerical=numerical_geometry_check()
    hole=sim.volume_manager.volumes['hole'];solid=hole.build_solid()
    lo,hi=g4.G4ThreeVector(),g4.G4ThreeVector();solid.BoundingLimits(lo,hi)
    bounds=np.array([[lo[i] for i in range(3)],[hi[i] for i in range(3)]])
    expected=np.array([[-RADIUS/math.cos(math.pi/6),-RADIUS,-LENGTH/2],
                       [RADIUS/math.cos(math.pi/6),RADIUS,LENGTH/2]])
    np.testing.assert_allclose(bounds,expected,rtol=0,atol=1e-7)
    volume=float(solid.GetCubicVolume())
    np.testing.assert_allclose(volume,2*math.sqrt(3)*RADIUS**2*LENGTH,rtol=1e-12)
    result=dict(numerical,actual_G4_bounds_mm=bounds.tolist(),actual_G4_volume_mm3=volume,
        apothem_semantics_verified=True,geometry_source_sha256=sha(Path(__file__)),
        role='numeric check inside formal transport; zero extra primary histories')
    save(Path(output)/'actual_hexagon_geometry.json',result)
    return result
