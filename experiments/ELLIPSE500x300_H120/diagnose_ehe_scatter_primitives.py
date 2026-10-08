"""Read-only scalar kinematics and angular-normalization diagnostics; no response generation."""
import math
import sys
sys.dont_write_bytecode=True
import numpy as np
from ehe_common import DATA,REPORT,digest,read,write


def kn(c,e):
    a=e/511.0*(1-c);b=1+c*c
    return b/((1+a)*(1+a))*(1+a*a/(b*(1+a)))


def frozen_normalization_cpu(e):
    f=np.float32;e=f(e);step=f(.01);pi=f(np.pi);total=f(0)
    for i in range(int(f(pi/step))+1):
        theta=f(i)*step;c=np.cos(theta);a=f(e/f(511))*f(f(1)-c);b=f(f(1)+c*c)
        value=f(f(b/f(f(1)+a))/f(f(1)+a))*f(f(1)+f(f(a*a/b)/f(f(1)+a)))
        total=f(total+f(f(value*np.sin(theta))*step))
    return float(total)


def acceptance(mean,lo,hi):
    sigma=.13*math.sqrt(511*mean)/2.35482
    lower=(lo-mean)/(math.sqrt(2)*sigma);upper=(hi-mean)/(math.sqrt(2)*sigma)
    if lower>=0:return .5*(math.erfc(lower)-math.erfc(upper))
    if upper<=0:return .5*(math.erfc(-upper)-math.erfc(-lower))
    return .5*(math.erf(upper)-math.erf(lower))


def audit():
    gpu=read(REPORT/'response_repair_freeze.json');cpu=read(REPORT/'transport_repair_freeze.json')
    gp=DATA/gpu['payload_dir'];cp=DATA/cpu['payload_dir']
    names=['engine/ScatterGen_RayTracing_CircularHole/scatter.cu','engine/common/detector_local_scatter.h',
        'engine/common/energy_window.h','engine/common/first_interaction.h']
    sources={}
    for n in names:
        assert digest(gp/n)==gpu['sha256'][n];sources['response:'+n]=digest(gp/n)
    for n in ['Geant4Code_EHE/src/EventAction.cc','Geant4Code_EHE/src/SteppingAction.cc']:
        assert digest(cp/n)==cpu['sha256'][n];sources['transport:'+n]=digest(cp/n)
    normalizations=[]
    for e in (218,440):
        references=[]
        for n in (256,512):
            c,w=np.polynomial.legendre.leggauss(n);references.append(float(np.dot(w,kn(c,e))))
        original=frozen_normalization_cpu(e)
        normalizations.append(dict(energy_keV=e,cpu_original_float32_theta_step_001=original,
            cosine_gauss256=references[0],cosine_gauss512=references[1],
            cpu_original_signed_relative_difference=(original-references[1])/references[1],
            reference256_512_signed_relative_difference=(references[0]-references[1])/references[1]))
    bounds=read(REPORT/'physical_source_window_basis_audit.json')['windows']['C440to218']['transport_bounds_keV']
    e0,e1,e2=440.,340.,222.
    cos1=1-511*(1/e1-1/e0);cos2=1-511*(1/e2-1/e1)
    assert -1<cos1<1 and -1<cos2<1 and e0-e2==218
    witness=dict(kind='Conditional kinematic example, not a measured Geant4 history or a contribution estimate',
        primary_energy_keV=e0,after_first_compton_keV=e1,after_second_compton_keV=e2,
        first_recoil_keV=e0-e1,second_recoil_keV=e1-e2,
        contained_same_crystal_cumulative_recoil_keV=e0-e2,
        first_angle_degrees=math.degrees(math.acos(cos1)),second_angle_degrees=math.degrees(math.acos(cos2)),
        conditional_cumulative218_window_probability=acceptance(218,*bounds),
        first_recoil100_window_probability=acceptance(100,*bounds),
        full440_window_probability=acceptance(440,*bounds),window_bounds_keV=bounds,
        assumptions='Both recoil electrons deposit their energy in the same NaI bin; final 222keV photon escapes that bin. This is an admissible energy history, not an observed event or an assigned geometry/path probability.',
        local_helper_coverage='The frozen local helper has no windowed term for its second-Compton partition. Its escape-recoil term applies the window to the first recoil; its photoelectric-followup term applies it to full primary energy.',
        whole_response_limitation='Full Scatter also has intercrystal and collimator terms. This local conditional example does not quantify their overlap, any missing global response fraction, or the cause of the observed 24.02% deficit.')
    result=dict(passed=True,scope='Read-only scalar arithmetic and conditional energy-history coverage only; original physical HOLD remains',
        scientific_status='HOLD',physical_gate_pass_claimed=False,science_job=1677211,
        angular_normalization=normalizations,conditional_energy_history=witness,
        frozen_source_sha256=sources,code_sha256=digest(__file__),physical_gate_sha256=digest(REPORT/'physical_gate.json'),
        no_physics_or_production_input_modified=True,no_observed_count_fit=True,no_response_or_simulation_executed=True,
        limitations='CPU arithmetic is not a capture of CUDA runtime values. Gaussian probabilities here are conditional reference arithmetic, not measured event fractions. Normalization checks do not bound finite surface/depth/collimator quadrature or missing histories.')
    write(REPORT/'physical_scatter_primitives_read_only.json',result)
    print('ANGULAR_NORMALIZATION_REFERENCE',normalizations)
    print('CONDITIONAL_KINEMATIC_WITNESS',witness)


if __name__=='__main__':audit()
