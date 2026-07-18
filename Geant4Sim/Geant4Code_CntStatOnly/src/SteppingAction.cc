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
/// \file hadronic/Hadr03/src/SteppingAction.cc
/// \brief Implementation of the SteppingAction class
//
// $Id: SteppingAction.cc 71404 2013-06-14 16:56:38Z maire $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


#include "DetectorConstruction.hh"
#include "SteppingAction.hh"
#include "Run.hh"
#include "EventAction.hh"

#include "G4RunManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"


#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <fstream>
using namespace std;


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(DetectorConstruction* det, EventAction* EvAct):
  G4UserSteppingAction(),
  fDetector(det),fEventAction(EvAct)
{
    //nx = det->Getnx();
    //ny = det->Getny();
    //nlayer = det->Getlayers();
    nScinNum = det->GetScinNum();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction()
{ }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* aStep)
{
  // A step's local energy deposit belongs to its pre-step volume. At a
  // geometry boundary the post-step touchable can already name the next
  // volume, which would assign the deposit to the wrong detector bin.
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();

  G4TouchableHandle theTouchable = preStepPoint->GetTouchableHandle();
  int numfinal = (theTouchable->GetCopyNumber()) - 1;
  G4double elocal = aStep->GetTotalEnergyDeposit();

  if(elocal > 0 && numfinal >= 0 && numfinal < nScinNum)
  {
    fEventAction->AddEnergy(numfinal, elocal);
  }

  /*
  if(elocal>0)
  {
    ofstream csv;
    csv.open("SteppingAction.csv",ios::app|ios::out);
    if (csv.bad()) G4cout<<" Csv failed"<<G4endl;
    else
    {
      // csv << (G4EventManager::GetEventManager())->GetConstCurrentEvent()->GetEventID() << ",";
      csv << aStep->GetTrack()->GetTrackID() << ",";
      csv << aStep->GetTrack()->GetParentID() << ",";
      csv << aStep->GetTrack()->GetParticleDefinition()->GetParticleName() << ",";
      csv << theTouchable->GetCopyNumber() << ",";
      // csv << Position.getX() << ",";
      // csv << Position.getY() << ",";
      // csv << Position.getZ() << endl;
      csv << aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << ",";
      csv << elocal << ",";
      csv.close();
    }
  }
  */
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
