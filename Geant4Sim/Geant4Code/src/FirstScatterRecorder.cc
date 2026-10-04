#include "FirstScatterRecorder.hh"
#include "DetectorConstruction.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"
#include <cstdlib>
#include <iomanip>
#include <stdexcept>

namespace {
std::string Env(const char* key,const char* fallback="unknown") {
  const char* value=std::getenv(key); return value ? value : fallback;
}
std::array<double,3> Vec(const G4ThreeVector& v,double unit=1) {
  return {{v.x()/unit,v.y()/unit,v.z()/unit}};
}
void WriteVec(std::ostream& out,const std::array<double,3>& v) {
  for (double x:v) out << ',' << x;
}
}
FirstScatterRecorder::FirstScatterRecorder(DetectorConstruction* det):detector(det) {
  policy=Env("JSCC_COMPTON_POLICY","legacy");
  if (policy!="legacy" && policy!="paired" && policy!="ideal_first_scatter_v2")
    throw std::runtime_error("Unknown JSCC_COMPTON_POLICY");
  dataset=Env("JSCC_DATASET"); worker=Env("JSCC_WORKER");
  seed=Env("JSCC_RANDOM_SEED"); view=Env("JSCC_VIEW");
  for(const auto& text:{dataset,worker,seed,view})
    if(text.find_first_of(",\r\n")!=std::string::npos) throw std::runtime_error("Unsafe event identity");
  summary.open("EventContract.csv",std::ios::out|std::ios::trunc);
  ideal.open("ListIdeal.csv",std::ios::out|std::ios::trunc);
  trace.open("PrimaryTrace.csv",std::ios::out|std::ios::trunc);
  if(!summary || !ideal || !trace) throw std::runtime_error("Cannot open event-contract outputs");
  summary << "dataset,worker,seed,view,event_id,legacy_row,ideal_row,legacy,ideal,reason,primary_mev,source_x,source_y,source_z,c1,c2,legacy_c1,legacy_c2,transfer_mev,true_e1,true_e2,measured_e1,measured_e2,first_origin_e1,foreign_e1,primary_interactions,first_process,first_pre_mev,first_post_mev,p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,in_x,in_y,in_z,out_x,out_y,out_z\n";
  trace << "dataset,worker,seed,view,event_id,ordinal,crystal,process,pre_mev,post_mev,x,y,z\n";
  summary << std::setprecision(17); trace << std::setprecision(17);
  // Match the legacy List writer's default six significant digits.
}
void FirstScatterRecorder::Begin(const G4Event* event) {
  state=jscc::FirstScatterState(); event_id=event->GetEventID();
  if(event->GetNumberOfPrimaryVertex()!=1 || event->GetPrimaryVertex()->GetNumberOfParticle()!=1)
    throw std::runtime_error("First-scatter contract requires one primary per event");
  state.source=Vec(event->GetPrimaryVertex()->GetPosition(),mm);
}
FirstScatterRecorder::~FirstScatterRecorder() {
  std::ofstream out("EmittedBins.csv");
  out << "radial_bin,axial_bin,primary_440\n";
  for(int r=0;r<3;++r) for(int z=0;z<3;++z)
    out << r << ',' << z << ',' << emitted440[r*3+z] << '\n';
}
void FirstScatterRecorder::Step(const G4Step* step) {
  const auto* track=step->GetTrack();
  const auto* pre=step->GetPreStepPoint(); const auto* post=step->GetPostStepPoint();
  const auto* physical=pre->GetPhysicalVolume();
  int crystal=-1;
  if(physical && detector->IsScintillator(physical->GetLogicalVolume())) {
    crystal=pre->GetTouchableHandle()->GetCopyNumber()-1;
    if(crystal<0 || crystal>=detector->GetScinNum()) throw std::runtime_error("Crystal mapping out of range");
  }
  const bool primary=track->GetParentID()==0 && track->GetParticleDefinition()->GetPDGEncoding()==22;
  int origin=0;
  if(primary) {
    if(track->GetCurrentStepNumber()==1) state.primary_mev=pre->GetKineticEnergy()/MeV;
    const auto* process=post->GetProcessDefinedStep();
    if(process && process->GetProcessType()!=fTransportation) {
      jscc::GammaInteraction interaction;
      interaction.crystal=crystal; interaction.process=process->GetProcessName();
      interaction.pre_mev=pre->GetKineticEnergy()/MeV; interaction.post_mev=post->GetKineticEnergy()/MeV;
      interaction.position=Vec(post->GetPosition(),mm);
      interaction.incoming=Vec(pre->GetMomentumDirection()); interaction.outgoing=Vec(post->GetMomentumDirection());
      state.primary.push_back(interaction);
      origin=state.primary.size()==1 ? 1 : 2;
    }
  } else {
    const auto* information=dynamic_cast<const FirstScatterLineage*>(track->GetUserInformation());
    if(information) origin=information->origin;
    else if(step->GetTotalEnergyDeposit()>0 || !step->GetSecondaryInCurrentStep()->empty())
      state.unknown_lineage=true;
  }
  const double deposit=step->GetTotalEnergyDeposit()/MeV;
  if(crystal>=0 && deposit>0) {
    state.deposited[crystal]+=deposit;
    if(origin==1) state.first_origin_deposited[crystal]+=deposit;
    else state.foreign_deposited[crystal]+=deposit;
  }
  for(const auto* secondary:*step->GetSecondaryInCurrentStep()) {
    if(secondary->GetUserInformation()) throw std::runtime_error("Secondary already has track information");
    const_cast<G4Track*>(secondary)->SetUserInformation(new FirstScatterLineage(origin));
    if(origin==0) state.unknown_lineage=true;
  }
}
jscc::FirstScatterDecision FirstScatterRecorder::End(const double* measured,int count,
                                                   bool legacy,int legacy_c1,int legacy_c2) {
  std::vector<int> hits;
  for(int i=0;i<count;++i) if(measured[i]>1*keV) hits.push_back(i);
  const auto decision=jscc::ClassifyFirstScatter(state,hits);
  if(std::abs(state.primary_mev-.440)<1e-9) {
    const double rho=std::hypot(state.source[0],state.source[1]+345.0)/255.0;
    const double z=std::abs(state.source[2]);
    const int rb=rho<=.5 ? 0 : (rho<=.85 ? 1 : 2);
    const int zb=z<=30 ? 0 : (z<=45 ? 1 : 2);
    ++emitted440[rb*3+zb];
  }
  const long long lr=legacy ? legacy_row++ : -1;
  const long long ir=decision.accepted ? ideal_row++ : -1;
  if(decision.accepted)
    ideal << decision.c1+1 << ',' << measured[decision.c1]/MeV << ','
          << decision.c2+1 << ',' << measured[decision.c2]/MeV << ",1\n";
  if(hits.size()>=2 || legacy || decision.accepted) {
    auto m=[&](int c){return c>=0 && c<count ? measured[c]/MeV : 0.0;};
    summary << dataset << ',' << worker << ',' << seed << ',' << view << ',' << event_id
      << ',' << lr << ',' << ir << ',' << legacy << ',' << decision.accepted << ',' << decision.reason
      << ',' << state.primary_mev;
    WriteVec(summary,state.source);
    summary << ',' << decision.c1+1 << ',' << decision.c2+1 << ',' << legacy_c1+1 << ',' << legacy_c2+1
      << ',' << decision.transfer_mev << ',' << jscc::Lookup(state.deposited,decision.c1)
      << ',' << jscc::Lookup(state.deposited,decision.c2) << ',' << m(decision.c1) << ',' << m(decision.c2)
      << ',' << jscc::Lookup(state.first_origin_deposited,decision.c1)
      << ',' << jscc::Lookup(state.foreign_deposited,decision.c1) << ',' << state.primary.size();
    jscc::GammaInteraction empty;
    const auto& first=state.primary.empty()?empty:state.primary[0];
    const auto& second=state.primary.size()<2?empty:state.primary[1];
    summary << ',' << (first.process.empty()?"none":first.process) << ',' << first.pre_mev << ',' << first.post_mev;
    WriteVec(summary,first.position); WriteVec(summary,second.position);
    WriteVec(summary,first.incoming); WriteVec(summary,first.outgoing); summary << '\n';
    // Fixed deterministic event sampling, capped at 100 traces per worker.
    if(event_id%1000==0 && trace_events<100) {
      ++trace_events;
      for(std::size_t i=0;i<state.primary.size();++i) {
        const auto& step=state.primary[i];
        trace << dataset << ',' << worker << ',' << seed << ',' << view << ',' << event_id
          << ',' << i << ',' << step.crystal+1 << ',' << step.process << ',' << step.pre_mev << ',' << step.post_mev;
        WriteVec(trace,step.position); trace << '\n';
      }
    }
  }
  if(!summary || !ideal || !trace) throw std::runtime_error("Event-contract write failed");
  return decision;
}
