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
/// \file hadronic/Hadr03/src/DetectorMessenger.cc
/// \brief Implementation of the DetectorMessenger class
//
// $Id: DetectorMessenger.cc 70755 2013-06-05 12:17:48Z ihrivnac $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"
#include "G4UIdirectory.hh"
#include "G4UIcommand.hh"
#include "G4UIparameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAnInteger.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::DetectorMessenger(DetectorConstruction * Det)
:G4UImessenger(), 
 fDetector(Det), fTestemDir(0), fDetDir(0),fpsMaterCmd(0),fscinMaterCmd(0),
 fSizeCmd(0),fGapSizeCmd(0),fLengthCmd(0)
{ 
  fTestemDir = new G4UIdirectory("/testhadr/");
  fTestemDir->SetGuidance("commands specific to this example");
  
  G4bool broadcast = false;
  fDetDir = new G4UIdirectory("/testhadr/det/",broadcast);
  fDetDir->SetGuidance("detector construction commands");
        
  fpsMaterCmd = new G4UIcmdWithAString("/testhadr/det/setPSMat",this);
  fpsMaterCmd->SetGuidance("Select material of the moderator.");
  fpsMaterCmd->SetParameterName("choice",false);
  fpsMaterCmd->AvailableForStates(G4State_PreInit,G4State_Idle);

  fscinMaterCmd = new G4UIcmdWithAString("/testhadr/det/setScinMat",this);
  fscinMaterCmd->SetGuidance("Select material of the detector.");
  fscinMaterCmd->SetParameterName("choice",false);
  fscinMaterCmd->AvailableForStates(G4State_PreInit,G4State_Idle);

  
  fSizeCmd = new G4UIcmdWithADoubleAndUnit("/testhadr/det/setSize",this);
  fSizeCmd->SetGuidance("Set size of one scintillator of the detector");
  fSizeCmd->SetParameterName("Size",false);
  fSizeCmd->SetRange("Size>0.");
  fSizeCmd->SetUnitCategory("Length");
  fSizeCmd->AvailableForStates(G4State_PreInit,G4State_Idle);

  fGapSizeCmd = new G4UIcmdWithADoubleAndUnit("/testhadr/det/setGapSize",this);
  fGapSizeCmd->SetGuidance("Set gap size of one scintillator of the detector");
  fGapSizeCmd->SetParameterName("GSize",false);
  fGapSizeCmd->SetRange("GSize>0.");
  fGapSizeCmd->SetUnitCategory("Length");
  fGapSizeCmd->AvailableForStates(G4State_PreInit,G4State_Idle);

  fLengthCmd = new G4UIcmdWithADoubleAndUnit("/testhadr/det/setLength",this);
  fLengthCmd->SetGuidance("Set length of the detector");
  fLengthCmd->SetParameterName("Length",false);
  fLengthCmd->SetRange("Length>0.");
  fLengthCmd->SetUnitCategory("Length");
  fLengthCmd->AvailableForStates(G4State_PreInit,G4State_Idle);




}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::~DetectorMessenger()
{
  delete fpsMaterCmd;
  delete fscinMaterCmd;
  delete fSizeCmd;
  delete fGapSizeCmd;
  delete fLengthCmd;
  delete fDetDir;
  delete fTestemDir;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == fpsMaterCmd )
   { fDetector->SetPSMaterial(newValue);}
  
  if( command == fscinMaterCmd)
   { fDetector->SetScinMaterial(newValue);}

  if( command == fSizeCmd )
   { fDetector->SetScinSize(fSizeCmd->GetNewDoubleValue(newValue));}

  if( command == fGapSizeCmd )
   { fDetector->SetGapXY(fGapSizeCmd->GetNewDoubleValue(newValue));}

  if( command == fLengthCmd )
  { fDetector->SetScinHeight(fLengthCmd->GetNewDoubleValue(newValue));}



}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
