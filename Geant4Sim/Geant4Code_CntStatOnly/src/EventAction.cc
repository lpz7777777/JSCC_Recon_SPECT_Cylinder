#include "EventAction.hh"

#include "DetectorConstruction.hh"
#include "Run.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>

EventAction::EventAction(DetectorConstruction* detector)
  : G4UserEventAction(),
    fDepositedEnergy(nullptr),
    fDetectorCount(detector->GetScinNum()),
    fEnergyResolutionRef(0.13),
    fEnergyResolutionRefEnergy(511 * keV)
{
  const G4double energy218 = 218 * keV;
  const G4double energy440 = 440 * keV;
  const G4double resolution218 =
      fEnergyResolutionRef * std::sqrt(fEnergyResolutionRefEnergy / energy218);
  const G4double resolution440 =
      fEnergyResolutionRef * std::sqrt(fEnergyResolutionRefEnergy / energy440);

  fWindow218Low = energy218 * (1.0 - resolution218 / 2.0);
  fWindow218High = energy218 * (1.0 + resolution218 / 2.0);
  fWindow440Low = energy440 * (1.0 - resolution440 / 2.0);
  fWindow440High = energy440 * (1.0 + resolution440 / 2.0);

  fDepositedEnergy = new G4double[fDetectorCount];
  std::fill_n(fDepositedEnergy, fDetectorCount, 0.0);
}

EventAction::~EventAction()
{
  delete[] fDepositedEnergy;
}

void EventAction::BeginOfEventAction(const G4Event*)
{
  std::fill_n(fDepositedEnergy, fDetectorCount, 0.0);
}

void EventAction::EndOfEventAction(const G4Event*)
{
  Run* run = static_cast<Run*>(
      G4RunManager::GetRunManager()->GetNonConstCurrentRun());

  for (int detectorIndex = 0; detectorIndex < fDetectorCount; ++detectorIndex)
  {
    G4double measuredEnergy = fDepositedEnergy[detectorIndex];
    if (measuredEnergy <= 0) continue;

    if (fEnergyResolutionRef > 0)
    {
      const G4double fwhm = fEnergyResolutionRef *
          std::sqrt(fEnergyResolutionRefEnergy / measuredEnergy);
      const G4double sigma = fwhm * measuredEnergy / 2.35482;
      measuredEnergy += CLHEP::RandGauss::shoot(0.0, sigma);
      if (measuredEnergy < 0) measuredEnergy = 0;
    }

    if (measuredEnergy >= fWindow440Low && measuredEnergy <= fWindow440High)
    {
      run->AddCnt440(detectorIndex);
    }
    if (measuredEnergy >= fWindow218Low && measuredEnergy <= fWindow218High)
    {
      run->AddCnt218(detectorIndex);
    }
  }
}
