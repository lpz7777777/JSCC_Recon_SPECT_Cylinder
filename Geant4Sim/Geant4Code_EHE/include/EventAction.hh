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
// $Id: B4aEventAction.hh 75215 2013-10-29 16:07:06Z gcosmo $
//
/// \file B4aEventAction.hh
/// \brief Definition of the B4aEventAction class

#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

/// Event action class
///
/// It defines data members to hold the energy deposit and track lengths
/// of charged particles in Absober and Gap layers:
/// - fEnergyAbs, fEnergyGap, fTrackLAbs, fTrackLGap
/// which are collected step by step via the functions
/// - AddAbs(), AddGap()

class DetectorConstruction;

class EventAction : public G4UserEventAction
{
  public:
    EventAction(DetectorConstruction*);
    virtual ~EventAction();

    virtual void BeginOfEventAction(const G4Event* event);
    virtual void EndOfEventAction(const G4Event* event);

    void AddEnergy(int number,G4double energy1){TempEnergy[number] += energy1;};

    // 能量分辨率高斯展宽设置
    void SetEnergyResolution(G4double res){fEnergyResolutionRef = res;};
    void SetEnergyResolutionRefEnergy(G4double e){fEnergyResolutionRefE = e;};
    G4double GetEnergyResolution(){return fEnergyResolutionRef;};
    G4double GetEnergyResolutionRefEnergy(){return fEnergyResolutionRefE;};
    // 双能量能窗设置
    void SetEnergy218(G4double e){fE218 = e;};
    void SetEnergy440(G4double e){fE440 = e;};

  private:
    G4double* TempEnergy;
    int nScinNum;
    G4double fEnergyResolutionRef;    // 基准 FWHM 能量分辨率（分数，如 0.13 = 13%）
    G4double fEnergyResolutionRefE;   // 基准能量（keV，如 511）
    G4double fE218;                   // 218keV 光电峰能量
    G4double fE440;                   // 440keV 光电峰能量
    G4double fWin218_lo, fWin218_hi;  // 218keV 能窗（自动由分辨率推算）
    G4double fWin440_lo, fWin440_hi;  // 440keV 能窗
};


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
