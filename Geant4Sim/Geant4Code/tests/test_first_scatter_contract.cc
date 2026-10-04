#include "FirstScatterContract.hh"
#include <iostream>
#include <stdexcept>

using namespace jscc;
GammaInteraction Interaction(int crystal,const std::string& process,double before,double after) {
  GammaInteraction i; i.crystal=crystal; i.process=process; i.pre_mev=before; i.post_mev=after; return i;
}
FirstScatterState Clean() {
  FirstScatterState s; s.primary_mev=.440;
  s.primary={Interaction(0,"compt",.440,.340),Interaction(1,"phot",.340,0)};
  s.deposited={{0,.100},{1,.340}}; s.first_origin_deposited={{0,.100}}; return s;
}
void Check(const FirstScatterState& s,const std::vector<int>& hits,bool expected,const char* name) {
  auto result=ClassifyFirstScatter(s,hits);
  if(result.accepted!=expected) throw std::runtime_error(std::string(name)+": "+result.reason);
  std::cout << name << ':' << result.reason << '\n';
}
int main() {
  auto s=Clean(); Check(s,{0,1},true,"clean");
  s=Clean(); s.primary[1]=Interaction(1,"compt",.340,.2); s.primary.push_back(Interaction(1,"phot",.2,0));
  Check(s,{0,1},true,"second_crystal_multistep");
  s=Clean(); s.primary.push_back(Interaction(0,"phot",.145877,0)); s.deposited[0]+=.145877;
  Check(s,{0,1},false,"return_photoelectric");
  s=Clean(); s.primary.insert(s.primary.begin(),Interaction(-1,"Rayl",.440,.440));
  Check(s,{0,1},false,"zero_deposit_prior_rayleigh");
  s=Clean(); Check(s,{0,1},true,"zero_local_deposit_recoil_contained");
  s=Clean(); s.primary.insert(s.primary.begin()+1,Interaction(0,"compt",.340,.250));
  Check(s,{0,1},false,"zero_local_deposit_first_of_two");
  Check(Clean(),{0,1,2},false,"third_measured_crystal");
  s=Clean(); s.deposited[0]=.095; s.first_origin_deposited[0]=.095;
  Check(s,{0,1},false,"secondary_escape");
  s=Clean(); s.foreign_deposited[0]=.001; s.deposited[0]+=.001;
  Check(s,{0,1},false,"secondary_return_contamination");
  s=Clean(); s.primary[0].crystal=10495; s.deposited[10495]=s.deposited[0]; s.deposited.erase(0);
  s.first_origin_deposited[10495]=s.first_origin_deposited[0]; s.first_origin_deposited.erase(0);
  Check(s,{10495,1},true,"refined_rear_layer_id");
  s=Clean(); s.primary_mev=.218; Check(s,{0,1},false,"wrong_primary_energy");
  s=Clean(); s.unknown_lineage=true; Check(s,{0,1},false,"missing_lineage");
  s=Clean(); s.deposited[0]+=.5e-6; Check(s,{0,1},true,"tolerance_inclusive_inside");
  s=Clean(); s.deposited[0]+=2e-6; Check(s,{0,1},false,"tolerance_outside");
  s=Clean(); s.primary.insert(s.primary.begin()+1,Interaction(-1,"compt",.340,.250));
  Check(s,{0,1},false,"scatter_between_crystals");
  s=Clean(); s.primary[1]=Interaction(1,"compt",.340,.1); s.deposited[1]=.240;
  Check(s,{0,1},true,"partial_second_absorption");
  std::cout << "FIRST_SCATTER_CONTRACT_TESTS_OK 16\n";
}
