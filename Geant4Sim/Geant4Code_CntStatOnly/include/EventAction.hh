#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

class DetectorConstruction;

class EventAction : public G4UserEventAction
{
  public:
    explicit EventAction(DetectorConstruction* detector);
    virtual ~EventAction();

    virtual void BeginOfEventAction(const G4Event* event);
    virtual void EndOfEventAction(const G4Event* event);

    void AddEnergy(int detectorIndex, G4double energy)
    {
      fDepositedEnergy[detectorIndex] += energy;
    }

  private:
    G4double* fDepositedEnergy;
    int fDetectorCount;
    G4double fEnergyResolutionRef;
    G4double fEnergyResolutionRefEnergy;
    G4double fWindow218Low;
    G4double fWindow218High;
    G4double fWindow440Low;
    G4double fWindow440High;
};

#endif
