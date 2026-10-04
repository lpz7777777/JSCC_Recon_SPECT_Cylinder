#ifndef JSCC_FIRST_SCATTER_RECORDER_HH
#define JSCC_FIRST_SCATTER_RECORDER_HH
#include "FirstScatterContract.hh"
#include "G4VUserTrackInformation.hh"
#include <fstream>
#include <memory>

class G4Step;
class G4Event;
class DetectorConstruction;

class FirstScatterLineage : public G4VUserTrackInformation {
 public:
  explicit FirstScatterLineage(int origin) : origin(origin) {}
  int origin; // 1=first primary interaction, 2=other primary interaction
  void Print() const override {}
};

class FirstScatterRecorder {
 public:
  explicit FirstScatterRecorder(DetectorConstruction* detector);
  ~FirstScatterRecorder();
  void Begin(const G4Event* event);
  void Step(const G4Step* step);
  jscc::FirstScatterDecision End(const double* measured, int count, bool legacy,
                                int legacy_c1, int legacy_c2);
  bool IdealOnly() const { return policy == "ideal_first_scatter_v2"; }
 private:
  DetectorConstruction* detector;
  jscc::FirstScatterState state;
  std::string policy, dataset, worker, seed, view;
  std::ofstream summary, ideal, trace;
  long long event_id=0, legacy_row=0, ideal_row=0, trace_events=0;
  std::array<long long,9> emitted440{{0,0,0,0,0,0,0,0,0}};
};
#endif
