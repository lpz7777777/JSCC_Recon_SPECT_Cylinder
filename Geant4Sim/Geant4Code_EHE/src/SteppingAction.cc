#include "SteppingAction.hh"

#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"

SteppingAction::SteppingAction(DetectorConstruction* detector, EventAction* eventAction)
  : G4UserSteppingAction(),
    fEventAction(eventAction),
    nScinNum(detector->GetScinNum())
{}

SteppingAction::~SteppingAction() = default;

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  const G4double depositedEnergy = step->GetTotalEnergyDeposit();
  // Energy deposition belongs to the pre-step volume. Using the post-step
  // touchable loses boundary-crossing deposits and can mistake another
  // volume's copy number for a detector bin.
  const auto* preStep = step->GetPreStepPoint();
  const auto* volume = preStep->GetPhysicalVolume();
  if (volume == nullptr || volume->GetName() != "Scin")
  {
    return;
  }

  const G4int detectorIndex = preStep->GetTouchableHandle()->GetCopyNumber() - 1;
  if (detectorIndex < 0 || detectorIndex >= nScinNum)
  {
    return;
  }
  if (depositedEnergy > 0.0)
  {
    fEventAction->AddEnergy(detectorIndex, depositedEnergy);
  }
}
