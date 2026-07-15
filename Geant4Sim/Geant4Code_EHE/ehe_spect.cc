#ifdef EHE_USE_UIVIS
#include "G4VisExecutive.hh"
#endif

#ifdef EHE_USE_UIVIS
#include "G4UIExecutive.hh"
#endif

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VModularPhysicsList.hh"
#include "QBBC.hh"
#include "Randomize.hh"
#include "SteppingVerbose.hh"

#include <cstdlib>

#ifdef _WIN32
#include <filesystem>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace
{
struct RandomSeed
{
  long value;
  bool isExplicit;
};

long CurrentProcessId()
{
#ifdef _WIN32
  return static_cast<long>(GetCurrentProcessId());
#else
  return static_cast<long>(getpid());
#endif
}

RandomSeed ResolveRandomSeed()
{
  if (const char* configured = std::getenv("EHE_RANDOM_SEED"))
  {
    char* end = nullptr;
    const long value = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' && value > 0)
    {
      return {value, true};
    }
    G4cerr << "Ignoring invalid EHE_RANDOM_SEED='" << configured << "'." << G4endl;
  }

  // Concurrent OS processes have distinct PIDs, matching the parallel seeding
  // scheme used by gamma01.cc.  Use EHE_RANDOM_SEED when reproducibility or
  // cross-host coordination requires a prescribed seed.
  return {CurrentProcessId(), false};
}

void SetInteractiveWorkingDirectory()
{
#ifdef _WIN32
  wchar_t executablePath[MAX_PATH];
  const DWORD length = GetModuleFileNameW(nullptr, executablePath, MAX_PATH);
  if (length > 0 && length < MAX_PATH)
  {
    std::error_code error;
    std::filesystem::current_path(
      std::filesystem::path(executablePath).parent_path(), error);
  }
#endif
}
}

int main(int argc, char** argv)
{
  if (argc == 1) SetInteractiveWorkingDirectory();

  G4Random::setTheEngine(new CLHEP::RanecuEngine);
  const RandomSeed randomSeed = ResolveRandomSeed();
  CLHEP::HepRandom::setTheSeed(randomSeed.value);
  G4cout << "EHE random seed: " << randomSeed.value
         << (randomSeed.isExplicit ? " (from EHE_RANDOM_SEED)" : " (from process ID)")
         << G4endl;

  G4VSteppingVerbose::SetInstance(new SteppingVerbose);
  auto* runManager = new G4RunManager;

  auto* detector = new DetectorConstruction;
  runManager->SetUserInitialization(detector);

  G4VModularPhysicsList* physics = new QBBC;
  physics->ReplacePhysics(new G4EmStandardPhysics_option4());
  runManager->SetUserInitialization(physics);
  runManager->SetUserInitialization(new ActionInitialization(detector));
  runManager->Initialize();

#ifdef EHE_USE_UIVIS
  auto* visManager = new G4VisExecutive;
  visManager->Initialize();
#endif

  auto* uiManager = G4UImanager::GetUIpointer();
  if (argc > 1)
  {
    uiManager->ApplyCommand(G4String("/control/execute ") + argv[1]);
  }
  else
  {
#ifdef EHE_USE_UIVIS
#ifdef G4UI_USE_WIN32
    auto* ui = new G4UIExecutive(argc, argv, "Win32");
#else
    auto* ui = new G4UIExecutive(argc, argv);
#endif
    uiManager->ApplyCommand("/control/execute vis.mac");
    ui->SessionStart();
    delete ui;
#else
    G4cerr << "A macro path is required in a build without Geant4 UI support." << G4endl;
#endif
  }

  delete runManager;
#ifdef EHE_USE_UIVIS
  delete visManager;
#endif
  return 0;
}
