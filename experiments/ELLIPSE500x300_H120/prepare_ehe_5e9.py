"""Prepare the exact approved H60 source and an isolated EHE source overlay."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import numpy as np
from ehe_common import *
from prepare_nema_simulation import prepare as prepare_source

def replace_once(path, old, new):
    s = path.read_text(encoding='utf-8')
    if s.count(old) != 1: raise ValueError('Source overlay anchor changed: ' + str(path))
    path.write_bytes(s.replace(old,new).encode())

def bounded_geometry_init(destination):
    # ValidateGeometry exhaustively checks hole boundaries/pitch and the touching
    # collimator/NaI planes. CSV-to-Params validation still checks every element.
    # Random Boolean surface sampling has pathological startup cost for 1250 holes.
    replace_once(destination/'src/DetectorConstruction.cc',
                 'collimatorLV, "EHECollimator", worldLV, false, 0, true);',
                 'collimatorLV, "EHECollimator", worldLV, false, 0, false);')
    main=destination/'ehe_spect.cc'
    replace_once(main,'  runManager->Initialize();',
                 '  const auto initStart=std::chrono::steady_clock::now();\n  runManager->Initialize();\n'
                 '  const double initSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-initStart).count();')
    replace_once(main,'  if (argc > 1)',
                 '  const auto beamStart=std::chrono::steady_clock::now();\n  if (argc > 1)')
    replace_once(main,'  delete runManager;',
                 '  const double beamSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-beamStart).count();\n'
                 '  std::ofstream timing("TransportTiming.json");\n'
                 '  timing<<"{\\\"initialization_seconds\\\":"<<initSeconds<<",\\\"beam_seconds\\\":"<<beamSeconds<<"}\\n";\n'
                 '  timing.close();\n  delete runManager;')
    text=main.read_text();main.write_text('#include <chrono>\n#include <fstream>\n'+text,encoding='utf-8',newline='\n')

def safe_multiunion(destination):
    # Geant4 11.1.0 uses size_t(size-1) even when the surface list is empty.
    # Compile the licensed, version-pinned implementation with just that guard
    # corrected into this executable; the installed toolkit remains read-only.
    shutil.copy2(HERE/'ehe_G4MultiUnion_11_1.cc',destination/'src/ehe_G4MultiUnion_11_1.cc')
    shutil.copy2(HERE/'ehe_GEANT4_LICENSE.txt',destination/'LICENSE.Geant4')
    link_multiunion(destination)
    source=destination/'src/DetectorConstruction.cc'
    replace_once(source,'  allHoles->Voxelize();','''  allHoles->Voxelize();
  int auditPoints=0;
  const auto audit=[&](const G4ThreeVector& point){
    Require(allHoles->Inside(point)==allHoles->InsideNoVoxels(point),
            "EHE_UNION", "Voxelized union differs from independent exhaustive solid oracle.");
    ++auditPoints;
  };
  audit(G4ThreeVector(0,-500*mm,0));
  audit(G4ThreeVector(500*mm,0,500*mm));
  for(const auto& h:holes)for(double dx:{0.0,1.249,1.251})for(double y:{-25.3,0.0,25.3})
    audit(G4ThreeVector((h.x+dx)*mm,y*mm,h.z*mm));
  std::ofstream unionAudit("EHE_MultiUnionAudit.json");
  unionAudit<<"{\\"passed\\":true,\\"points\\":"<<auditPoints<<",\\"holes\\":1250,\\"empty_candidates_guard\\":true}\\n";
  unionAudit.close();''')

def link_multiunion(destination):
    # This entry point has an explicit source list, not a src/*.cc glob.
    replace_once(destination/'CMakeLists.txt','  src/SteppingVerbose.cc\n)',
                 '  src/SteppingVerbose.cc\n  src/ehe_G4MultiUnion_11_1.cc\n)')

def overlay(destination):
    base = ROOT/'Geant4Sim/Geant4Code_EHE'
    shutil.copytree(base,destination,ignore=shutil.ignore_patterns('build*','out','CntStat*','EHE_Geometry*','EHE_DetectorGeometry.csv','EHE_CollimatorHoles.csv','*.log','__pycache__'))
    source_identity = hashes(destination)
    for rel in ('src/PrimaryGeneratorAction.cc','include/PrimaryGeneratorAction.hh'):
        source=ROOT/'Geant4Sim/Geant4Code'/rel
        shutil.copy2(source,destination/rel)
        source_identity['JSCC/'+rel]=digest(source)
    replace_once(destination/'include/PrimaryGeneratorAction.hh','fSourceCenterY = -245','fSourceCenterY = -345')
    replace_once(destination/'include/DetectorConstruction.hh','kFovCenterY = -245.0','kFovCenterY = -345.0')
    replace_once(destination/'include/DetectorConstruction.hh','kCommonFrontFaceDistance = 198.5','kCommonFrontFaceDistance = 298.5')
    replace_once(destination/'src/DetectorConstruction.cc','Collimator front face is not 198.5 mm','Collimator front face is not 298.5 mm')
    replace_once(destination/'validate_against_params.py','FOV_CENTER_Y_MM = -245.0','FOV_CENTER_Y_MM = -345.0')
    bounded_geometry_init(destination)
    safe_multiunion(destination)
    # Read-only diagnostics use no random draws and retain the original window decisions.
    (destination/'include/Run.hh').write_text('''#ifndef Run_h
#define Run_h 1
#include "G4Run.hh"
#include "globals.hh"
#include <array>
#include <vector>
class DetectorConstruction;
class Run : public G4Run {
public:
 explicit Run(DetectorConstruction*);
 DetectorConstruction* getDetector(){return detector;}
 void AddCnt218(int i){counts[0][i]++;}
 void AddCnt440(int i){counts[1][i]++;}
 G4long GetCnt218(int i){return counts[0][i];}
 G4long GetCnt440(int i){return counts[1][i];}
 void AddTagged(int window,int primary,int i){tagged[2*window+primary][i]++;}
 G4long GetTagged(int window,int primary,int i) const{return tagged[2*window+primary][i];}
 void Merge(const G4Run*) override;
private:
 DetectorConstruction* detector;
 std::array<std::vector<G4long>,2> counts;
 std::array<std::vector<G4long>,4> tagged;
};
#endif
''',encoding='utf-8')
    (destination/'src/Run.cc').write_text('''#include "Run.hh"
#include "DetectorConstruction.hh"
Run::Run(DetectorConstruction* d):detector(d){
 for(auto& v:counts)v.resize(d->GetScinNum());
 for(auto& v:tagged)v.resize(d->GetScinNum());
}
void Run::Merge(const G4Run* r){
 const auto* o=static_cast<const Run*>(r);
 for(int j=0;j<2;++j)for(size_t i=0;i<counts[j].size();++i)counts[j][i]+=o->counts[j][i];
 for(int j=0;j<4;++j)for(size_t i=0;i<tagged[j].size();++i)tagged[j][i]+=o->tagged[j][i];
 G4Run::Merge(r);
}
''',encoding='utf-8')
    event=destination/'src/EventAction.cc'
    replace_once(event,'#include "G4Event.hh"','#include "G4Event.hh"\n#include "G4PrimaryVertex.hh"\n#include "G4PrimaryParticle.hh"')
    replace_once(event,'const G4Event* /*event*/','const G4Event* event')
    replace_once(event,'  // Count every NaI detector bin', '''  const auto primaryKeV=event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()/keV;
  const int primary=(std::abs(primaryKeV-218.0)<0.001)?0:1;
  if(std::abs(primaryKeV-218.0)>=0.001 && std::abs(primaryKeV-440.0)>=0.001)
    G4Exception("EventAction", "EHE_PRIMARY", FatalException, "Only 218/440 primaries permitted.");
  // Count every NaI detector bin''')
    replace_once(event,'run->AddCnt440(i);','run->AddCnt440(i);\n      run->AddTagged(1,primary,i);')
    replace_once(event,'run->AddCnt218(i);','run->AddCnt218(i);\n      run->AddTagged(0,primary,i);')
    action=destination/'src/RunAction.cc'
    replace_once(action,'  // save Rndm status','  fPrimary->ResetPrimaryCounts();\n  // save Rndm status')
    text=action.read_text(encoding='utf-8').replace('const G4int count =','const G4long count =')
    anchor='  G4int nbOfEvents = aRun->GetNumberOfEvent();'
    if text.count(anchor)!=1: raise ValueError('RunAction source changed')
    extra='''
  for(int w=0;w<2;++w)for(int p=0;p<2;++p){
    const auto name=std::string("CntStat_")+(w==0?"218":"440")+"_from"+(p==0?"218":"440")+".csv";
    std::ofstream tagged(name,std::ios::out|std::ios::app);
    if(!tagged)G4Exception("RunAction","EHE_WRITE",FatalException,"Cannot write tagged counts.");
    for(int i=0;i<nScinNum;++i)tagged<<fRun->GetTagged(w,p,i)<<(i+1<nScinNum?",":"\\n");
  }
  std::ofstream summary("TransportSummary.json",std::ios::out|std::ios::trunc);
  if(!summary)G4Exception("RunAction","EHE_WRITE",FatalException,"Cannot write primary receipt.");
  summary<<"{\\"primary_events\\":"<<nbOfEvents<<",\\"primary_counts\\":["
         <<fPrimary->GetPrimary218()<<","<<fPrimary->GetPrimary440()<<","<<fPrimary->GetPrimaryOther()
         <<"],\\"detector_bins\\":"<<nScinNum<<"}\\n";
'''
    action.write_bytes(text.replace(anchor,anchor+extra).encode())
    write(destination/'overlay_manifest.json',dict(base_source_sha256=source_identity,
        source_sampling='exact baseline weighted-cuboid sampler, unchanged random draws',
        changes=['source reference -345/front298.5','primary-labelled non-random diagnostic counters',
                 'deterministic geometry validation replaces random Boolean overlap surface sampling; no geometry change',
                 'separate initialization/beam timers, no random draws'],
        files=hashes(destination)))

def prepare():
    if DATA.exists(): raise FileExistsError('Experiment already prepared; use status')
    old=read(HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/formal_summary.json')
    if not old.get('passed') or old.get('job')!=1669255: raise ValueError('Accepted JSCC reference required')
    if digest(TRUTH)!=read(TRUTH_META)['truth_sha256']: raise ValueError('Truth changed')
    # The seed allocation is independent of every existing preparation. Candidate range
    # is searched in the entire tracked/small JSON registration, not just source code.
    candidate=31100101
    registered=set()
    for p in HERE.rglob('*.json'):
        if p.stat().st_size>4_000_000: continue
        try: value=read(p)
        except (ValueError,UnicodeError): continue
        def visit(v):
            if isinstance(v,dict):
                for k,item in v.items():
                    if k=='seed' and isinstance(item,int): registered.add(item)
                    visit(item)
            elif isinstance(v,list):
                for item in v:visit(item)
        visit(value)
    while any(s in registered for s in range(candidate,candidate+200)):candidate+=1000
    DATA.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    prepare_source(DATA/'simulation',5_000_000_000,10,'5e9',candidate)
    sim=read(DATA/'simulation/jobs.json');sim.update(study=STUDY,hardware='EHE Pb/NaI',jscc_reference_job=1669255)
    write(DATA/'simulation/jobs.json',sim)
    payload=DATA/'payload';payload.mkdir()
    overlay(payload/'Geant4Code_EHE')
    shutil.copy2(GEOMETRY,payload/'whole_geometry.npz')
    with np.load(GEOMETRY) as g:
        if len(g['active_indices'])!=78920 or len(g['coordinates_mm'])!=132040:raise ValueError('Whole geometry differs')
    old_params=ENGINE/'FileGenerater_3D_Unified/output'
    parameter_sha={}
    for response,folder in zip(RESPONSES,('EHE_PbNaI_218keV','EHE_PbNaI_440keV','EHE_PbNaI_440keV_to_218keVwin')):
        target=payload/'params'/response;target.mkdir(parents=True)
        for p in sorted((old_params/folder).glob('Params_*.dat')):shutil.copy2(p,target/p.name)
        if len(list(target.glob('*.dat')))!=4:raise ValueError('Four EHE Params required')
        np.array([85,85,40,6,6,3,1,0,0,0,0,323.75],'<f4').tofile(target/'Params_Image.dat')
        det=np.fromfile(target/'Params_Detector.dat','<f4')
        col=np.fromfile(target/'Params_Collimator.dat','<f4')
        if det[0]!=2312 or col[10]!=1250:raise ValueError('EHE dimensions changed')
        parameter_sha[response]=hashes(target)
    # Preserve raw authoritative truth bytes, including archive metadata.
    shutil.copy2(TRUTH,payload/'truth_3mm.npz');shutil.copy2(TRUTH_META,payload/'truth_manifest.json')
    coords=np.load(TRUTH);xx,yy=np.meshgrid(coords['x_mm'],coords['y_mm'])
    rows=[]
    for view in range(20):
        theta=view*2*np.pi/20;projected=xx*np.cos(theta)+yy*np.sin(theta)
        mask=np.abs(projected)>136
        rows.append(dict(view=view+1,angle_deg=view*18,
            **{f'geometric_outside_detector_fraction_{e}':float(np.sum(coords[f'activity_{e}_zyx']*mask)/np.sum(coords[f'activity_{e}_zyx'])) for e in (218,440)}))
    write(REPORT/'projection_truncation.json',dict(rows=rows,
        meaning='orthographic source activity outside detector face; not the measured photon rejection probability',
        detector_face_mm=[272,136]))
    config=dict(study=STUDY,total_primary_photons=5_000_000_000,views=20,workers=200,
        photons_per_worker=25_000_000,seed_base=candidate,detector_bins=2312,hole_count=1250,
        source_center_mm=[0,-345,0],front_distance_mm=298.5,params_local_origin_mm=323.75,
        iterations=200,save_step=10,channels=list(CHANNELS),responses=list(RESPONSES),
        gamma_yields={'218':.114,'440':.259},truth_sha256=digest(TRUTH),geometry_sha256=digest(GEOMETRY),
        simulation_manifest_sha256=digest(DATA/'simulation/jobs.json'),params_sha256=parameter_sha,
        primary_count_gate=True,physical_hold_relative=.10,physical_hold_sigma=3,
        physical_adequate_min_counts=100,resource_fraction_limit=.8,
        cross_background_source='EHE 440 single final200; JSCC frozen 440 single final10000',
        source_voxel_basis='3mm cuboids; no object attenuation',device_response_cache='none',
        jscc_reference_job=1669255,baseline_modified=False)
    write(payload/'config.json',config)
    for name in ('ehe_common.py','prepare_ehe_5e9.py','ehe_5e9_workflow.py','run_ehe_worker.py','ehe_gpu_pipeline.py','run_ehe_reconstruction.py','verify_ehe.py','compare_ehe_5e9.py','ehe_reference_evidence.py','test_ehe_5e9.py','ehe_pe_v4_overlay.py','ehe_pe_chord_test.cu','validate_ehe_pe_v4.py'):
        shutil.copy2(HERE/name,payload/name)
    for name in ('torch_active_operator.py','single_checkpoint_mlem.py'):
        shutil.copy2(HERE/name,payload/name)
    # Freeze only current engine sources/includes, never its outputs or old executables.
    for name in ('common','physics_data','PEGen_RayTracing_CircularHole','ScatterGen_RayTracing_CircularHole'):
        for p in (ENGINE/name).rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.h','.cuh','.cu','.cpp'):
                out=payload/'engine'/p.relative_to(ENGINE);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    from ehe_pe_v4_overlay import overlay as overlay_pe
    overlay_pe(ENGINE/'PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu',payload/'engine/PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu')
    files=hashes(payload);key=__import__('hashlib').sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'freeze.json',dict(study=STUDY,release_key=key,sha256=files,config_sha256=digest(payload/'config.json')))
    print('EHE_PREPARED',key,'seed_base',candidate)

if __name__=='__main__':prepare()
