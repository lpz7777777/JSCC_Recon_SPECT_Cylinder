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
/// \file electromagnetic/TestEm11/include/Run.hh
/// \brief Definition of the Run class
//
// $Id: Run.hh 71375 2013-06-14 07:39:33Z maire $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef Run_h
#define Run_h 1

#include "G4Run.hh"
#include "G4VProcess.hh"
#include "globals.hh"
#include <array>
#include <map>
#include <vector>

class DetectorConstruction;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class Run : public G4Run
{
  public:
    Run(DetectorConstruction*);
    ~Run();

  public:
	  DetectorConstruction* getDetector(){return fDetector;};
    void AddCnt218(int number){LocalCnt_218[number]+=1;};
    void AddCnt440(int number){LocalCnt_440[number]+=1;};
    void AddCnt218From218(int number){LocalCnt218From218[number]+=1;};
    void AddCnt218From440(int number){LocalCnt218From440[number]+=1;};
    void AddCnt440From440(int number){LocalCnt440From440[number]+=1;};
    void AddCnt218From440FirstCrystal(int number, int comptonCount)
    {
      LocalCnt218From440FirstCrystal[number] += 1;
      const int category = comptonCount <= 0 ? 0 : (comptonCount == 1 ? 1 : 2);
      LocalCnt218From440FirstCrystalCompton[category][number] += 1;
    };
    void AddCnt218From440OtherCrystal(int number)
    {
      LocalCnt218From440OtherCrystal[number] += 1;
    };
    void AddCnt218From440HitMultiplicity(int number, int hitMultiplicity)
    {
      const int category = hitMultiplicity <= 1 ? 0 : (hitMultiplicity == 2 ? 1 : 2);
      LocalCnt218From440HitMultiplicity[category][number] += 1;
    };
    void AddPrimary218(){PrimaryCount218++;};
    void AddPrimary440(){PrimaryCount440++;};
    void AddPrimaryOther(){PrimaryCountOther++;};
    void AddEnergySpectrum(G4double Energy);
    void AddEventType(int Type){EventType[Type]++;};
    void AddList(G4double Energy1, G4double Energy2, int Flag1, int Flag2, int Flag);

    int GetCnt218(int number){return LocalCnt_218[number];};
    int GetCnt440(int number){return LocalCnt_440[number];};
    int GetCnt218From218(int number){return LocalCnt218From218[number];};
    int GetCnt218From440(int number){return LocalCnt218From440[number];};
    int GetCnt440From440(int number){return LocalCnt440From440[number];};
    int GetCnt218From440FirstCrystal(int number){return LocalCnt218From440FirstCrystal[number];};
    int GetCnt218From440OtherCrystal(int number){return LocalCnt218From440OtherCrystal[number];};
    int GetCnt218From440Hit1(int number){return LocalCnt218From440HitMultiplicity[0][number];};
    int GetCnt218From440Hit2(int number){return LocalCnt218From440HitMultiplicity[1][number];};
    int GetCnt218From440Hit3Plus(int number){return LocalCnt218From440HitMultiplicity[2][number];};
    int GetCnt218From440FirstCrystalCompton0(int number){return LocalCnt218From440FirstCrystalCompton[0][number];};
    int GetCnt218From440FirstCrystalCompton1(int number){return LocalCnt218From440FirstCrystalCompton[1][number];};
    int GetCnt218From440FirstCrystalCompton2Plus(int number){return LocalCnt218From440FirstCrystalCompton[2][number];};
    G4long GetPrimaryCount218() const {return PrimaryCount218;};
    G4long GetPrimaryCount440() const {return PrimaryCount440;};
    G4long GetPrimaryCountOther() const {return PrimaryCountOther;};
    int GetnergySpectrum(int energy){return EnergySpectrum[energy];};
    int GetEventType(int Type){return EventType[Type];};
    int GetTotalCount() const {return static_cast<int>(List.size());};
    G4double GetList(int i, int j) const {return List.at(i).at(j);};

    virtual void Merge(const G4Run*);


  private:
    DetectorConstruction* fDetector;
    int* LocalCnt_218;
    int* LocalCnt_440;
    int* LocalCnt218From218;
    int* LocalCnt218From440;
    int* LocalCnt440From440;
    int* LocalCnt218From440FirstCrystal;
    int* LocalCnt218From440OtherCrystal;
    std::array<int*, 3> LocalCnt218From440HitMultiplicity;
    std::array<int*, 3> LocalCnt218From440FirstCrystalCompton;
    G4long PrimaryCount218;
    G4long PrimaryCount440;
    G4long PrimaryCountOther;
    int* EnergySpectrum; 
    int nx, ny, nlayer;
    int nScinNum;
    int* EventType;
    std::vector<std::array<G4double, 5> > List;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

