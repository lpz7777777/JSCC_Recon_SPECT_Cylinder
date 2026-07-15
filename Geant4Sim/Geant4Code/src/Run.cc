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
/// \file electromagnetic/TestEm11/src/Run.cc
/// \brief Implementation of the Run class
//
// $Id: Run.cc 71376 2013-06-14 07:44:50Z maire $··
// 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "Run.hh"
#include "DetectorConstruction.hh"

#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Run::Run(DetectorConstruction* det):
  G4Run(),
  fDetector(det), LocalCnt_218(0), LocalCnt_440(0), EnergySpectrum(0), EventType(0)
{
  nx = det->Getnx();
  ny = det->Getny();
  nlayer = det->Getlayers();
  nScinNum = det->GetScinNum();
  LocalCnt_218 = new int[nScinNum];
  LocalCnt_440 = new int[nScinNum];
  for(int i=0; i<nScinNum; i++)
  {
    LocalCnt_218[i] = 0;
    LocalCnt_440[i] = 0;
  }

  /*
  EnergySpectrum = new int[3000];
  for(int i=0; i<3000; i++)
  {
    EnergySpectrum[i] = 0;
  }
  */

  /*
  EventType = new int[8];
  for(int i=0; i<8; i++)
  {
    EventType[i] = 0;
  }
  */

  // Grow only with accepted Compton events. The old implementation allocated
  // ten million separate rows up front and could write past that fixed limit.
  List.reserve(100000);
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Run::~Run()
{
  delete[] LocalCnt_218;
  delete[] LocalCnt_440;
  // delete[] EnergySpectrum;
  // delete[] EventType;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Run::Merge(const G4Run* run)
{
  const Run* localRun = static_cast<const Run*>(run);
  // counts (两个能量的单光子投影分别合并)
  for(int i=0; i<nScinNum; i++)
  {
    LocalCnt_218[i] += localRun->LocalCnt_218[i];
    LocalCnt_440[i] += localRun->LocalCnt_440[i];
  }
  List.insert(List.end(), localRun->List.begin(), localRun->List.end());
  G4Run::Merge(run);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Run::AddEnergySpectrum(G4double Energy)
{
  int i = round(Energy / keV);
  EnergySpectrum[i]++;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

//void Run::AddList(G4double Energy1, G4double Energy2, int Flag1, int Flag2, int Flag, G4ThreeVector Position1, G4ThreeVector Position2)
void Run::AddList(G4double Energy1, G4double Energy2, int Flag1, int Flag2, int Flag)
{
  List.push_back({{
    static_cast<G4double>(Flag1 + 1),
    Energy1,
    static_cast<G4double>(Flag2 + 1),
    Energy2,
    static_cast<G4double>(Flag)
  }});
  /*
  Position columns can be added by extending the std::array row type.
  */
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
