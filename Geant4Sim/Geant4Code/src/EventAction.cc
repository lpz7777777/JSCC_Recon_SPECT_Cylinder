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
// $Id: B4aEventAction.cc 75604 2013-11-04 13:17:26Z gcosmo $
//
/// \file B4aEventAction.cc
/// \brief Implementation of the B4aEventAction class

#include <math.h>
#include "EventAction.hh"
#include "DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4Event.hh"
#include "Run.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "Randomize.hh"
#include <iomanip>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
using namespace std;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::EventAction(DetectorConstruction* det):
  G4UserEventAction(),
  fDetector(det), TempEnergy(0), TempEnergy_Real(0), Scin_CopyNum(0), NumCompt(0), NumPhot(0), TempEnergy_CrystalTotal(0), TempEnergy_CrystalTotal_Real(0), TempEnergyPos_CrystalTotal(0), OutCopyNum(0), NumScinEachCrystal(0)
{
  Threshold = 1 * keV;
  // 能量分辨率默认值：511keV 处 FWHM=13%，其他能量按 R∝1/√E 标度
  fEnergyResolutionRef = 0.13;
  fEnergyResolutionRefE = 511 * keV;
  // 双能量光电峰及能窗（能窗 = E × (1 ± res(E)/2)，res(E) 由基准标度）
  fE218 = 218 * keV;
  fE440 = 440 * keV;
  G4double res218 = fEnergyResolutionRef * std::sqrt(fEnergyResolutionRefE / fE218);
  G4double res440 = fEnergyResolutionRef * std::sqrt(fEnergyResolutionRefE / fE440);
  fWin218_lo = fE218 * (1.0 - res218 / 2.0);
  fWin218_hi = fE218 * (1.0 + res218 / 2.0);
  fWin440_lo = fE440 * (1.0 - res440 / 2.0);
  fWin440_hi = fE440 * (1.0 + res440 / 2.0);
  // nx = det->Getnx();
  // ny = det->Getny();
  // nlayer = det->Getlayers();
  nScinNum = det->GetScinNum();

  TempEnergy = new G4double[nScinNum];
  for(int i=0; i<nScinNum; i++)
  {
    TempEnergy[i] = 0;
  }
  Scin_CopyNum = -2;
  NumCompt = 0;
  NumPhot = 0;
  Flag_Compt = 0;
  //Position1 = G4ThreeVector(0,0,0);
  //Position2 = G4ThreeVector(0,0,0);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::~EventAction()
{
  delete[] TempEnergy;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::BeginOfEventAction(const G4Event*)
{
  // initialisation per event
  for(int i=0; i<nScinNum; i++)
  {
    TempEnergy[i]=0;
  }
  Scin_CopyNum = -2;
  NumCompt = 0;
  NumPhot = 0;
  Flag_Compt = 0;
  //Position1 = G4ThreeVector(0,0,0);
  //Position2 = G4ThreeVector(0,0,0);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event* /*event*/)
{
  // ====== 能量分辨率高斯展宽 ======
  // 对每个有能量沉积的晶体，模拟探测器的能量分辨率噪声：
  //   FWHM(E) = res_ref × √(E_ref / E)      （闪烁体 R∝1/√E 标度）
  //   σ = FWHM / 2.35482
  //   E_measured = E_deposit + Gaus(0, σ)
  // 展宽后的 TempEnergy 用于后续 CntStat/List 判定和输出
  if (fEnergyResolutionRef > 0)
  {
    for (int i = 0; i < nScinNum; i++)
    {
      if (TempEnergy[i] > 0)
      {
        G4double E = TempEnergy[i];
        G4double fwhm = fEnergyResolutionRef * std::sqrt(fEnergyResolutionRefE / E);
        G4double sigma = fwhm * E / 2.35482;
        TempEnergy[i] = E + CLHEP::RandGauss::shoot(0.0, sigma);
        if (TempEnergy[i] < 0) TempEnergy[i] = 0;  // 截断负值
      }
    }
  }

  int Flag1, Flag2, Flag3, FirstScinCount;
  Flag1 = -1;
  Flag2 = -1;
  Flag3 = -1;
  FirstScinCount = 0;
  Run* run = static_cast<Run*>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());

  // ====== 找能量最高的晶体 ======
  int   maxIdx = -1;
  G4double maxE  = 0;
  for(int i=0; i<nScinNum; i++)
  {
    if(TempEnergy[i] > maxE)
    {
      maxE = TempEnergy[i];
      maxIdx = i;
    }
  }

  // ====== 单光子事件：最高能量落在能窗内 ======
  if(maxIdx >= 0)
  {
    if(maxE >= fWin440_lo && maxE <= fWin440_hi)
    {
      run->AddCnt440(maxIdx);
      return;  // 单光子事件，结束
    }
    if(maxE >= fWin218_lo && maxE <= fWin218_hi)
    {
      run->AddCnt218(maxIdx);
      return;  // 单光子事件，结束
    }
  }

  // ====== 康普顿事件：非单光子，检查 2 晶体 + 1 次散射 ======
  for(int i=0; i<nScinNum; i++)
  {
    if(TempEnergy[i] > Threshold)
    {
      if (Flag1==-1)
      {
        Flag1 = i;
      }
      else if(Flag2 == -1)
      {
        Flag2 = i;
      }
      else
      {
        Flag3 = 1;
        break;
      }

      if(i==Scin_CopyNum)
      {
        FirstScinCount=1;
      }
    }
  }

  if (Flag2 != -1 && Flag3 == -1 && FirstScinCount==1 && NumCompt==1 && NumPhot==0 && Flag_Compt==1)
  {
    if(Flag1 == Scin_CopyNum)
    {
      run->AddList(TempEnergy[Flag1], TempEnergy[Flag2], Flag1, Flag2, 1);
    }
    else
    {
      run->AddList(TempEnergy[Flag2], TempEnergy[Flag1], Flag2, Flag1, 1);
    }
  }

  /*
  if (Flag2 != -1 && Flag3 == -1)
  {
    if(FirstScinCount==1 && NumCompt==1 && NumPhot==0 && Flag_Compt==1)
    {
      Flag = 1;
    }
    else
    {
      Flag = 2;
    }

    if(Flag1 == Scin_CopyNum)
    {
      run->AddList(TempEnergy[Flag1], TempEnergy[Flag2], Flag1, Flag2, Flag);
    }
    else if(Flag2 == Scin_CopyNum)
    {
      run->AddList(TempEnergy[Flag2], TempEnergy[Flag1], Flag2, Flag1, Flag);
    }
  }
  */
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......