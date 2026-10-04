#ifndef JSCC_FIRST_SCATTER_CONTRACT_HH
#define JSCC_FIRST_SCATTER_CONTRACT_HH

// Geant4-independent classification, also exercised by the C++ unit tests.
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace jscc {
struct GammaInteraction {
  int crystal = -1;                       // 0-based, -1 means non-scintillator
  std::string process;
  double pre_mev = 0, post_mev = 0;
  std::array<double,3> position{{0,0,0}};
  std::array<double,3> incoming{{0,0,0}}, outgoing{{0,0,0}};
};
struct FirstScatterState {
  double primary_mev = 0;
  std::array<double,3> source{{0,0,0}};
  std::vector<GammaInteraction> primary;
  std::map<int,double> deposited, first_origin_deposited, foreign_deposited;
  bool unknown_lineage = false;
};
struct FirstScatterDecision {
  bool accepted = false;
  std::string reason;
  int c1 = -1, c2 = -1;
  double transfer_mev = 0, tolerance_mev = 0;
};
inline double Lookup(const std::map<int,double>& values, int id) {
  const auto found = values.find(id);
  return found == values.end() ? 0.0 : found->second;
}
inline FirstScatterDecision ClassifyFirstScatter(
    const FirstScatterState& state, const std::vector<int>& measured_hits) {
  FirstScatterDecision result;
  auto reject = [&](const char* why) { result.reason = why; return result; };
  if (!std::isfinite(state.primary_mev)) return reject("nonfinite_primary");
  if (std::abs(state.primary_mev - .440) > 1e-9) return reject("not_440_primary");
  if (state.primary.empty()) return reject("no_primary_interaction");
  const auto& first = state.primary.front();
  result.c1 = first.crystal;
  result.transfer_mev = first.pre_mev - first.post_mev;
  result.tolerance_mev = std::max(1e-6, 1e-5 * result.transfer_mev);
  if (first.crystal < 0 || first.process != "compt") return reject("first_not_crystal_compton");
  if (!std::isfinite(result.transfer_mev) || result.transfer_mev <= 0)
    return reject("invalid_first_transfer");
  if (state.unknown_lineage) return reject("unknown_secondary_lineage");
  // The very next discrete primary interaction must be in a different crystal.
  if (state.primary.size() < 2) return reject("no_second_crystal_interaction");
  result.c2 = state.primary[1].crystal;
  if (result.c2 < 0 || result.c2 == result.c1) return reject("intermediate_or_same_crystal_interaction");
  for (std::size_t i=1; i<state.primary.size(); ++i)
    if (state.primary[i].crystal == result.c1) return reject("return_to_first_crystal");
  if (Lookup(state.foreign_deposited, result.c1) > result.tolerance_mev)
    return reject("foreign_energy_in_first_crystal");
  if (std::abs(Lookup(state.deposited, result.c1)-result.transfer_mev) > result.tolerance_mev)
    return reject("first_energy_not_closed");
  if (std::abs(Lookup(state.first_origin_deposited, result.c1)-result.transfer_mev) > result.tolerance_mev)
    return reject("first_origin_energy_not_closed");
  if (measured_hits.size() != 2) return reject("not_two_measured_crystals");
  if (std::find(measured_hits.begin(),measured_hits.end(),result.c1)==measured_hits.end() ||
      std::find(measured_hits.begin(),measured_hits.end(),result.c2)==measured_hits.end())
    return reject("measured_pair_differs_from_primary");
  result.accepted = true; result.reason = "accepted";
  return result;
}
}
#endif
