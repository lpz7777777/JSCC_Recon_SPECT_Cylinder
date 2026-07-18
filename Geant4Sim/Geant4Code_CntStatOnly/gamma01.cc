//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file hadronic/Hadr03/Hadr03.cc
/// \brief Main program of the hadronic/Hadr03 example
//
//
// $Id: TestEm1.cc,v 1.16 2010-04-06 11:11:24 maire Exp $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifdef GAMMA01_USE_UIVIS
#include "G4VisExecutive.hh"
#endif

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VModularPhysicsList.hh"

#ifdef GAMMA01_USE_UIVIS
#include "G4UIExecutive.hh"
#endif
#include "G4EmStandardPhysics_option4.hh"

#include "ActionInitialization.hh"
#include "Randomize.hh"
#include "DetectorConstruction.hh"
#include "SteppingVerbose.hh"

#include "QBBC.hh"
#include "FTFP_BERT.hh"
#include "QGSP_BIC_HP.hh"
#include "Randomize.hh"
#include "G4Exception.hh"
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#include <filesystem>
#else
#include <unistd.h>
#endif


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

namespace
{
  constexpr std::uint64_t kMaximumRanecuSeed = 2147483562ULL;

  std::uint64_t Mix64(std::uint64_t value)
  {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  std::uint64_t HashText(const char* text)
  {
    std::uint64_t hash = 1469598103934665603ULL;
    if (!text) return hash;
    while (*text)
    {
      hash ^= static_cast<unsigned char>(*text++);
      hash *= 1099511628211ULL;
    }
    return hash;
  }

  std::uint64_t ProcessId()
  {
#ifdef _WIN32
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
  }

  G4long BuildRandomSeed()
  {
    const char* configured = std::getenv("JSCC_RANDOM_SEED");
    if (configured && configured[0] != '\0')
    {
      errno = 0;
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(configured, &end, 10);
      if (errno != 0 || end == configured || *end != '\0' ||
          parsed == 0 || parsed > kMaximumRanecuSeed)
      {
        G4Exception("BuildRandomSeed", "InvalidSeed", FatalException,
                    "JSCC_RANDOM_SEED must be an integer in [1, 2147483562].");
        return 1;
      }
      return static_cast<G4long>(parsed);
    }

    const auto now = std::chrono::high_resolution_clock::now()
                         .time_since_epoch().count();
    std::uint64_t material = static_cast<std::uint64_t>(now);
    material ^= Mix64(ProcessId());
    material ^= Mix64(HashText(std::getenv("SLURM_JOB_ID")));
    material ^= Mix64(HashText(std::getenv("SLURM_ARRAY_JOB_ID")));
    material ^= Mix64(HashText(std::getenv("SLURM_ARRAY_TASK_ID")));
    return static_cast<G4long>(1ULL + Mix64(material) % kMaximumRanecuSeed);
  }

  const char* EnvironmentOrDash(const char* name)
  {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' ? value : "-";
  }

  // Explorer does not guarantee that a double-clicked console application
  // starts with its executable directory as the current working directory.
  // The interactive mode needs CrystalMatrix.txt and vis.mac beside gamma01.
  // Batch mode deliberately keeps the caller's working directory so its CSV
  // results continue to be written into each dedicated run directory.
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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int main(int argc,char** argv)
{
  if (argc == 1) SetInteractiveWorkingDirectory();

  // Choose the Random engine
  G4Random::setTheEngine(new CLHEP::RanecuEngine);
  const G4long seed = BuildRandomSeed();
  CLHEP::HepRandom::setTheSeed(seed);
  G4cout << "Random seed initialization: seed=" << seed
         << ", pid=" << ProcessId()
         << ", SLURM_JOB_ID=" << EnvironmentOrDash("SLURM_JOB_ID")
         << ", SLURM_ARRAY_JOB_ID=" << EnvironmentOrDash("SLURM_ARRAY_JOB_ID")
         << ", SLURM_ARRAY_TASK_ID=" << EnvironmentOrDash("SLURM_ARRAY_TASK_ID")
         << G4endl;


  // Construct the default run manager
  G4VSteppingVerbose::SetInstance(new SteppingVerbose);
  G4RunManager* runManager = new G4RunManager;


  // Set mandatory initialization classes
  // Detector construction
  DetectorConstruction* det= new DetectorConstruction;
  runManager->SetUserInitialization(det);

  // Physics list
  G4VModularPhysicsList* phys = new QBBC;
  phys->ReplacePhysics(new G4EmStandardPhysics_option4());
  runManager->SetUserInitialization(phys);

  // User action initialization
  runManager->SetUserInitialization(new ActionInitialization(det));

  // Initialize G4 kernel
  runManager->Initialize();

#ifdef GAMMA01_USE_UIVIS
  // Initialize visualization
  G4VisManager* visManager = new G4VisExecutive;
  visManager->Initialize();
#endif


  // Get the pointer to the User Interface manager
  G4UImanager* UI = G4UImanager::GetUIpointer();
  if (argc!=1)
  // batch mode
    {
      G4String command = "/control/execute ";
      G4String fileName = argv[1];
      UI->ApplyCommand(command + fileName);
    }
  else
  //define visualization and UI terminal for interactive mode
    {
#ifdef GAMMA01_USE_UIVIS
       // Force Geant4's native Win32 command window.  Relying on automatic
       // selection may pick a non-interactive fallback when launched from
       // Explorer instead of an existing terminal.  Other platforms retain
       // Geant4's normal automatic session selection.
#ifdef G4UI_USE_WIN32
       G4UIExecutive * ui = new G4UIExecutive(argc,argv,"Win32");
#else
       G4UIExecutive * ui = new G4UIExecutive(argc,argv);
#endif
       UI->ApplyCommand("/control/execute vis.mac");
       ui->SessionStart();
       delete ui;
#else
       G4cerr << "Interactive UI/visualisation was not built. "
              << "Reconfigure with -DWITH_GEANT4_UIVIS=ON or pass a macro file."
              << G4endl;
#endif
    }


  // job termination
  delete runManager;
#ifdef GAMMA01_USE_UIVIS
  delete visManager;
#endif

  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
