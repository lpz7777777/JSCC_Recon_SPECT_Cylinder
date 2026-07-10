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

#ifdef G4VIS_USE
#include "G4VisExecutive.hh"
#endif

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VModularPhysicsList.hh"

#ifdef G4UI_USE
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
#include "time.h"


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int main(int argc,char** argv)
{
  // Choose the Random engine
  G4Random::setTheEngine(new CLHEP::RanecuEngine);
  G4long seed = time(NULL);
  CLHEP::HepRandom::setTheSeed(seed);


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

#ifdef G4VIS_USE
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
#ifdef G4UI_USE
      G4UIExecutive * ui = new G4UIExecutive(argc,argv);
#ifdef G4VIS_USE
      UI->ApplyCommand("/control/execute vis.mac");
#endif
      ui->SessionStart();
      delete ui;
#endif
    }


  // job termination
  delete runManager;
#ifdef G4VIS_USE
  delete visManager;
#endif

  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
