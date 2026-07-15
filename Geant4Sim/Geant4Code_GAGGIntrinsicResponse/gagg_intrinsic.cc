#include "FTFP_BERT.hh"
#include "G4Box.hh"
#include "G4EmParameters.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4Event.hh"
#include "G4Exception.hh"
#include "G4Gamma.hh"
#include "G4GenericMessenger.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysicalConstants.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#ifdef G4MULTITHREADED
#include "G4MTRunManager.hh"
#endif
#include "G4StateManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4UserEventAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VProcess.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ios.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

namespace
{
enum class InteractionKind
{
    None,
    Photoelectric,
    Compton,
    Other
};

InteractionKind ClassifyProcess(const G4String& name)
{
    if (name == "phot") return InteractionKind::Photoelectric;
    if (name == "compt") return InteractionKind::Compton;
    return InteractionKind::Other;
}

G4double Ratio(G4long numerator, G4long denominator)
{
    if (denominator <= 0) return std::numeric_limits<G4double>::quiet_NaN();
    return static_cast<G4double>(numerator) / static_cast<G4double>(denominator);
}
}

class StudyDetectorConstruction final : public G4VUserDetectorConstruction
{
public:
    StudyDetectorConstruction()
        : fWidth(3.0 * mm),
          fThickness(3.0 * mm),
          fHeight(3.0 * mm),
          fEnergy(218.0 * keV),
          fFullEnergyTolerance(1.0 * eV),
          fFaceMargin(1.0 * um),
          fOutput("gagg_intrinsic_response.csv"),
          fCrystalLogical(nullptr),
          fMessenger(new G4GenericMessenger(this, "/study/", "GAGG intrinsic-response study"))
    {
        auto& width = fMessenger->DeclarePropertyWithUnit(
            "crystalWidth", "mm", fWidth, "Crystal X size");
        auto& thickness = fMessenger->DeclarePropertyWithUnit(
            "crystalThickness", "mm", fThickness, "Crystal Y size (beam axis)");
        auto& height = fMessenger->DeclarePropertyWithUnit(
            "crystalHeight", "mm", fHeight, "Crystal Z size");
        auto& energy = fMessenger->DeclarePropertyWithUnit(
            "energy", "keV", fEnergy, "Primary gamma energy");
        auto& tolerance = fMessenger->DeclarePropertyWithUnit(
            "fullEnergyTolerance", "eV", fFullEnergyTolerance,
            "Allowed missing energy for physical full-energy containment");
        auto& margin = fMessenger->DeclarePropertyWithUnit(
            "faceMargin", "um", fFaceMargin,
            "Unsampled margin at each edge of the illuminated face");
        auto& output = fMessenger->DeclareProperty(
            "output", fOutput, "Output CSV path");
        width.SetStates(G4State_PreInit);
        thickness.SetStates(G4State_PreInit);
        height.SetStates(G4State_PreInit);
        energy.SetStates(G4State_PreInit, G4State_Idle);
        tolerance.SetStates(G4State_PreInit, G4State_Idle);
        margin.SetStates(G4State_PreInit, G4State_Idle);
        output.SetStates(G4State_PreInit, G4State_Idle);
    }

    ~StudyDetectorConstruction() override
    {
        delete fMessenger;
    }

    G4VPhysicalVolume* Construct() override
    {
        if (!(fWidth > 0.0) || !(fThickness > 0.0) || !(fHeight > 0.0))
        {
            G4Exception("StudyDetectorConstruction::Construct", "InvalidCrystalSize",
                FatalException, "All crystal dimensions must be positive.");
        }

        G4NistManager* nist = G4NistManager::Instance();
        G4Material* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
        G4Element* gd = nist->FindOrBuildElement("Gd");
        G4Element* al = nist->FindOrBuildElement("Al");
        G4Element* ga = nist->FindOrBuildElement("Ga");
        G4Element* oxygen = nist->FindOrBuildElement("O");
        G4Element* ce = nist->FindOrBuildElement("Ce");

        G4Material* base = G4Material::GetMaterial("GAGGBase", false);
        if (base == nullptr)
        {
            base = new G4Material("GAGGBase", 6.6 * g / cm3, 4);
            base->AddElement(gd, 3);
            base->AddElement(al, 2);
            base->AddElement(ga, 3);
            base->AddElement(oxygen, 12);
        }
        G4Material* gagg = G4Material::GetMaterial("GAGG", false);
        if (gagg == nullptr)
        {
            gagg = new G4Material("GAGG", 6.6 * g / cm3, 2);
            gagg->AddMaterial(base, 99.0 * perCent);
            gagg->AddElement(ce, 1.0 * perCent);
        }

        const G4double worldHalf = 10.0 * std::max({fWidth, fThickness, fHeight});
        auto* worldSolid = new G4Box("WorldSolid", worldHalf, worldHalf, worldHalf);
        auto* worldLogical = new G4LogicalVolume(worldSolid, vacuum, "WorldLogical");
        auto* worldPhysical = new G4PVPlacement(
            nullptr, G4ThreeVector(), worldLogical, "World", nullptr, false, 0, true);

        auto* crystalSolid = new G4Box(
            "CrystalSolid", 0.5 * fWidth, 0.5 * fThickness, 0.5 * fHeight);
        fCrystalLogical = new G4LogicalVolume(crystalSolid, gagg, "CrystalLogical");
        new G4PVPlacement(nullptr, G4ThreeVector(), fCrystalLogical,
            "Crystal", worldLogical, false, 0, true);
        return worldPhysical;
    }

    G4double GetWidth() const { return fWidth; }
    G4double GetThickness() const { return fThickness; }
    G4double GetHeight() const { return fHeight; }
    G4double GetEnergy() const { return fEnergy; }
    G4double GetFullEnergyTolerance() const { return fFullEnergyTolerance; }
    G4double GetFaceMargin() const { return fFaceMargin; }
    const G4String& GetOutput() const { return fOutput; }
    G4LogicalVolume* GetCrystalLogical() const { return fCrystalLogical; }

private:
    G4double fWidth;
    G4double fThickness;
    G4double fHeight;
    G4double fEnergy;
    G4double fFullEnergyTolerance;
    G4double fFaceMargin;
    G4String fOutput;
    G4LogicalVolume* fCrystalLogical;
    G4GenericMessenger* fMessenger;
};

struct EventSummary
{
    G4bool entered = false;
    G4bool contained = false;
    InteractionKind first = InteractionKind::None;
    InteractionKind second = InteractionKind::None;
    G4bool eventualPhotoelectric = false;
};

class ResponseRun final : public G4Run
{
public:
    void Record(const EventSummary& event)
    {
        if (event.entered) ++entered;
        if (event.contained) ++fullEnergyAll;
        if (event.first == InteractionKind::Photoelectric)
        {
            ++firstPE;
            if (event.contained) ++firstPEContained;
        }
        else if (event.first == InteractionKind::Compton)
        {
            ++firstCompton;
            if (event.second == InteractionKind::Photoelectric)
            {
                ++firstComptonSecondPE;
                if (event.contained) ++firstComptonSecondPEContained;
            }
            if (event.eventualPhotoelectric)
            {
                ++firstComptonEventualPE;
                if (event.contained) ++firstComptonEventualPEContained;
            }
        }
        else if (event.first == InteractionKind::Other)
        {
            ++firstOther;
        }
        else
        {
            ++noInteraction;
        }
    }

    void Merge(const G4Run* run) override
    {
        const auto* other = static_cast<const ResponseRun*>(run);
        entered += other->entered;
        fullEnergyAll += other->fullEnergyAll;
        firstPE += other->firstPE;
        firstPEContained += other->firstPEContained;
        firstCompton += other->firstCompton;
        firstComptonSecondPE += other->firstComptonSecondPE;
        firstComptonSecondPEContained += other->firstComptonSecondPEContained;
        firstComptonEventualPE += other->firstComptonEventualPE;
        firstComptonEventualPEContained += other->firstComptonEventualPEContained;
        firstOther += other->firstOther;
        noInteraction += other->noInteraction;
        G4Run::Merge(run);
    }

    G4long entered = 0;
    G4long fullEnergyAll = 0;
    G4long firstPE = 0;
    G4long firstPEContained = 0;
    G4long firstCompton = 0;
    G4long firstComptonSecondPE = 0;
    G4long firstComptonSecondPEContained = 0;
    G4long firstComptonEventualPE = 0;
    G4long firstComptonEventualPEContained = 0;
    G4long firstOther = 0;
    G4long noInteraction = 0;
};

class StudyPrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
public:
    explicit StudyPrimaryGeneratorAction(const StudyDetectorConstruction* detector)
        : fDetector(detector), fGun(new G4ParticleGun(1))
    {
        fGun->SetParticleDefinition(G4Gamma::GammaDefinition());
        fGun->SetParticleMomentumDirection(G4ThreeVector(0.0, 1.0, 0.0));
    }

    ~StudyPrimaryGeneratorAction() override { delete fGun; }

    void GeneratePrimaries(G4Event* event) override
    {
        const G4double marginX = std::min(fDetector->GetFaceMargin(),
            0.49 * fDetector->GetWidth());
        const G4double marginZ = std::min(fDetector->GetFaceMargin(),
            0.49 * fDetector->GetHeight());
        const G4double spanX = fDetector->GetWidth() - 2.0 * marginX;
        const G4double spanZ = fDetector->GetHeight() - 2.0 * marginZ;
        const G4double x = (G4UniformRand() - 0.5) * spanX;
        const G4double z = (G4UniformRand() - 0.5) * spanZ;
        const G4double y = -0.5 * fDetector->GetThickness() - 1.0 * um;
        fGun->SetParticleEnergy(fDetector->GetEnergy());
        fGun->SetParticlePosition(G4ThreeVector(x, y, z));
        fGun->GeneratePrimaryVertex(event);
    }

private:
    const StudyDetectorConstruction* fDetector;
    G4ParticleGun* fGun;
};

class StudyEventAction final : public G4UserEventAction
{
public:
    explicit StudyEventAction(const StudyDetectorConstruction* detector)
        : fDetector(detector) {}

    void BeginOfEventAction(const G4Event*) override
    {
        fEnergyDeposit = 0.0;
        fEntered = false;
        fFirst = InteractionKind::None;
        fSecond = InteractionKind::None;
        fPrimaryInteractionCount = 0;
        fEventualPhotoelectric = false;
    }

    void EndOfEventAction(const G4Event*) override
    {
        EventSummary summary;
        summary.entered = fEntered;
        summary.contained = fEnergyDeposit
            >= fDetector->GetEnergy() - fDetector->GetFullEnergyTolerance();
        summary.first = fFirst;
        summary.second = fSecond;
        summary.eventualPhotoelectric = fEventualPhotoelectric;
        auto* run = static_cast<ResponseRun*>(
            G4RunManager::GetRunManager()->GetNonConstCurrentRun());
        run->Record(summary);
    }

    void AddEnergyDeposit(G4double value) { fEnergyDeposit += value; }
    void MarkEntered() { fEntered = true; }

    void RecordPrimaryInteraction(const G4String& processName)
    {
        const InteractionKind kind = ClassifyProcess(processName);
        ++fPrimaryInteractionCount;
        if (fPrimaryInteractionCount == 1) fFirst = kind;
        if (fPrimaryInteractionCount == 2) fSecond = kind;
        if (fFirst == InteractionKind::Compton
            && kind == InteractionKind::Photoelectric
            && fPrimaryInteractionCount >= 2)
            fEventualPhotoelectric = true;
    }

private:
    const StudyDetectorConstruction* fDetector;
    G4double fEnergyDeposit = 0.0;
    G4bool fEntered = false;
    InteractionKind fFirst = InteractionKind::None;
    InteractionKind fSecond = InteractionKind::None;
    G4int fPrimaryInteractionCount = 0;
    G4bool fEventualPhotoelectric = false;
};

class StudySteppingAction final : public G4UserSteppingAction
{
public:
    StudySteppingAction(
        const StudyDetectorConstruction* detector,
        StudyEventAction* eventAction)
        : fDetector(detector), fEventAction(eventAction) {}

    void UserSteppingAction(const G4Step* step) override
    {
        const G4StepPoint* pre = step->GetPreStepPoint();
        if (pre->GetTouchableHandle()->GetVolume() == nullptr) return;
        if (pre->GetTouchableHandle()->GetVolume()->GetLogicalVolume()
            != fDetector->GetCrystalLogical())
            return;

        fEventAction->AddEnergyDeposit(step->GetTotalEnergyDeposit());
        const G4Track* track = step->GetTrack();
        if (track->GetTrackID() != 1
            || track->GetDefinition() != G4Gamma::GammaDefinition())
            return;

        fEventAction->MarkEntered();
        const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
        if (process == nullptr || process->GetProcessName() == "Transportation") return;
        fEventAction->RecordPrimaryInteraction(process->GetProcessName());
    }

private:
    const StudyDetectorConstruction* fDetector;
    StudyEventAction* fEventAction;
};

class StudyRunAction final : public G4UserRunAction
{
public:
    explicit StudyRunAction(const StudyDetectorConstruction* detector)
        : fDetector(detector) {}

    G4Run* GenerateRun() override { return new ResponseRun; }

    void EndOfRunAction(const G4Run* genericRun) override
    {
        if (!IsMaster()) return;
        const auto* run = static_cast<const ResponseRun*>(genericRun);
        std::ofstream output(fDetector->GetOutput());
        if (!output)
        {
            G4Exception("StudyRunAction::EndOfRunAction", "OutputOpenFailed",
                FatalException, fDetector->GetOutput().c_str());
        }
        output << "width_mm,thickness_mm,height_mm,energy_keV,tolerance_eV,events,"
            "entered,full_energy_all,first_pe,first_pe_contained,first_pe_containment,"
            "first_compton,first_compton_second_pe,first_compton_second_pe_contained,"
            "first_compton_second_pe_containment,first_compton_eventual_pe,"
            "first_compton_eventual_pe_contained,first_compton_eventual_pe_containment,"
            "first_other,no_interaction\n";
        output << std::setprecision(12)
            << fDetector->GetWidth() / mm << ','
            << fDetector->GetThickness() / mm << ','
            << fDetector->GetHeight() / mm << ','
            << fDetector->GetEnergy() / keV << ','
            << fDetector->GetFullEnergyTolerance() / eV << ','
            << run->GetNumberOfEvent() << ','
            << run->entered << ','
            << run->fullEnergyAll << ','
            << run->firstPE << ','
            << run->firstPEContained << ','
            << Ratio(run->firstPEContained, run->firstPE) << ','
            << run->firstCompton << ','
            << run->firstComptonSecondPE << ','
            << run->firstComptonSecondPEContained << ','
            << Ratio(run->firstComptonSecondPEContained,
                run->firstComptonSecondPE) << ','
            << run->firstComptonEventualPE << ','
            << run->firstComptonEventualPEContained << ','
            << Ratio(run->firstComptonEventualPEContained,
                run->firstComptonEventualPE) << ','
            << run->firstOther << ','
            << run->noInteraction << '\n';
        G4cout << "Wrote intrinsic-response result to "
            << fDetector->GetOutput() << G4endl;
    }

private:
    const StudyDetectorConstruction* fDetector;
};

class StudyActionInitialization final : public G4VUserActionInitialization
{
public:
    explicit StudyActionInitialization(const StudyDetectorConstruction* detector)
        : fDetector(detector) {}

    void BuildForMaster() const override
    {
        SetUserAction(new StudyRunAction(fDetector));
    }

    void Build() const override
    {
        SetUserAction(new StudyPrimaryGeneratorAction(fDetector));
        SetUserAction(new StudyRunAction(fDetector));
        auto* eventAction = new StudyEventAction(fDetector);
        SetUserAction(eventAction);
        SetUserAction(new StudySteppingAction(fDetector, eventAction));
    }

private:
    const StudyDetectorConstruction* fDetector;
};

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        G4cerr << "Usage: gagg_intrinsic macro.mac [threads]" << G4endl;
        return 2;
    }

    G4RunManager* runManager = nullptr;
#ifdef G4MULTITHREADED
    auto* mtRunManager = new G4MTRunManager;
    const G4int threads = argc == 3 ? std::max(1, std::atoi(argv[2])) : 1;
    mtRunManager->SetNumberOfThreads(threads);
    runManager = mtRunManager;
    G4cout << "Using " << threads << " Geant4 worker thread(s)." << G4endl;
#else
    runManager = new G4RunManager;
    if (argc == 3 && std::atoi(argv[2]) != 1)
        G4cout << "This Geant4 build is sequential; the thread argument is ignored."
            << G4endl;
#endif
    auto* detector = new StudyDetectorConstruction;
    runManager->SetUserInitialization(detector);

    auto* physics = new FTFP_BERT;
    physics->ReplacePhysics(new G4EmStandardPhysics_option4());
    physics->SetDefaultCutValue(0.1 * um);
    runManager->SetUserInitialization(physics);
    G4EmParameters::Instance()->SetFluo(true);
    G4EmParameters::Instance()->SetAuger(true);
    G4EmParameters::Instance()->SetPixe(true);
    runManager->SetUserInitialization(new StudyActionInitialization(detector));

    const G4String command = "/control/execute ";
    const G4int status = G4UImanager::GetUIpointer()->ApplyCommand(command + argv[1]);
    delete runManager;
    return status == 0 ? 0 : 1;
}
