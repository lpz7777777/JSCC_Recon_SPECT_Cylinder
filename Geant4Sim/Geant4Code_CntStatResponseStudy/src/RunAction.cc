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
/// \file hadronic/Hadr03/src/RunAction.cc
/// \brief Implementation of the RunAction class
//
// $Id: RunAction.cc 70756 2013-06-05 12:20:06Z ihrivnac $
// 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "RunAction.hh"

#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "Run.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

#include "Randomize.hh"
#include <iomanip>

#include <iostream>
#include <fstream>
using namespace std;

namespace
{
void WriteDetectorCounts(const char* filename, Run* run, int nScinNum,
                         int (Run::*getter)(int))
{
  ofstream csv(filename, ios::app|ios::out);
  if (!csv.is_open())
  {
    G4cout << " Failed to open " << filename << G4endl;
    return;
  }

  G4long total = 0;
  for(int k=0; k<nScinNum; k++)
  {
    const G4int count = (run->*getter)(k);
    total += count;
    csv << count;
    if(k<nScinNum-1) csv << ",";
    else csv << endl;
  }
  G4cout << "... wrote " << filename << ", accepted crystal counts = "
         << total << "." << G4endl;
}

void WritePrimaryCount(const char* filename, G4long count)
{
  ofstream csv(filename, ios::app|ios::out);
  if (!csv.is_open())
  {
    G4cout << " Failed to open " << filename << G4endl;
    return;
  }
  csv << count << endl;
  G4cout << "... wrote " << filename << ", primary events = "
         << count << "." << G4endl;
}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction(DetectorConstruction* det, PrimaryGeneratorAction* prim)
  : G4UserRunAction(),
    fDetector(det), fPrimary(prim), fRun(0)
{
  // Book predefined histograms
  nScinNum = det->GetScinNum();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::~RunAction()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4Run* RunAction::GenerateRun()
{ 
  fRun = new Run(fDetector); 
  return fRun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run*)
{    
  // save Rndm status
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);
  G4Random::showEngineStatus();
       
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run* aRun)
{
  G4int nbOfEvents = aRun->GetNumberOfEvent();

  if (fPrimary && nbOfEvents) 
  { 
    G4cout << "\n The run generated " << nbOfEvents
           << " primary events from the configured GPS source mixture."
           << G4endl;

    //Write CntStat_218 into a csv file
    ofstream csv;
    csv.open("CntStat_218.csv", ios::app|ios::out);
    if (!csv.is_open())
    {
      G4cout<<" Csv failed"<<G4endl;
    }
    else
    {
      G4cout << "... write Csv File : CntStat_218.csv ";
      G4long totalCnt218 = 0;
      for(int k=0; k<nScinNum; k++)
      {
        const G4int count = fRun->GetCnt218(k);
        totalCnt218 += count;
        csv << count;
        if(k<nScinNum-1) csv << ",";
        else csv << endl;
      }
      csv.close();
      G4cout << "- done, accepted events = " << totalCnt218 << "." << G4endl;
    }

    //Write CntStat_440 into a csv file
    csv.open("CntStat_440.csv", ios::app|ios::out);
    if (!csv.is_open())
    {
      G4cout<<" Csv failed"<<G4endl;
    }
    else
    {
      G4cout << "... write Csv File : CntStat_440.csv ";
      G4long totalCnt440 = 0;
      for(int k=0; k<nScinNum; k++)
      {
        const G4int count = fRun->GetCnt440(k);
        totalCnt440 += count;
        csv << count;
        if(k<nScinNum-1) csv << ",";
        else csv << endl;
      }
      csv.close();
      G4cout << "- done, accepted events = " << totalCnt440 << "." << G4endl;
    }

    WriteDetectorCounts("CntStat218_From218.csv", fRun, nScinNum,
                        &Run::GetCnt218From218);
    WriteDetectorCounts("CntStat218_From440.csv", fRun, nScinNum,
                        &Run::GetCnt218From440);
    WriteDetectorCounts("CntStat440_From440.csv", fRun, nScinNum,
                        &Run::GetCnt440From440);
    WriteDetectorCounts("CntStat218_From440_FirstCrystal.csv", fRun, nScinNum,
                        &Run::GetCnt218From440FirstCrystal);
    WriteDetectorCounts("CntStat218_From440_OtherCrystal.csv", fRun, nScinNum,
                        &Run::GetCnt218From440OtherCrystal);
    WriteDetectorCounts("CntStat218_From440_Hit1.csv", fRun, nScinNum,
                        &Run::GetCnt218From440Hit1);
    WriteDetectorCounts("CntStat218_From440_Hit2.csv", fRun, nScinNum,
                        &Run::GetCnt218From440Hit2);
    WriteDetectorCounts("CntStat218_From440_Hit3Plus.csv", fRun, nScinNum,
                        &Run::GetCnt218From440Hit3Plus);
    WriteDetectorCounts("CntStat218_From440_FirstCrystal_Compton0.csv", fRun,
                        nScinNum, &Run::GetCnt218From440FirstCrystalCompton0);
    WriteDetectorCounts("CntStat218_From440_FirstCrystal_Compton1.csv", fRun,
                        nScinNum, &Run::GetCnt218From440FirstCrystalCompton1);
    WriteDetectorCounts("CntStat218_From440_FirstCrystal_Compton2Plus.csv", fRun,
                        nScinNum, &Run::GetCnt218From440FirstCrystalCompton2Plus);
    WritePrimaryCount("PrimaryCount218.csv", fRun->GetPrimaryCount218());
    WritePrimaryCount("PrimaryCount440.csv", fRun->GetPrimaryCount440());
    WritePrimaryCount("PrimaryCountOther.csv", fRun->GetPrimaryCountOther());

    /*
    csv.open("EnergySpectrum.csv", ios::app|ios::out);
    if (csv.bad())
    {
      G4cout<<" Csv failed"<<G4endl;
    }
    else
    {
      G4cout << "... write Csv File : EnergySpectrum.csv ";
      for(int k=0; k<3000; k++)
      {
        csv << fRun->GetnergySpectrum(k);
        if(k<2999)
        {
          csv << ",";
        }
        else
        {
          csv << endl;
        }
      }
      csv.close();
      G4cout << "- done." << G4endl;
    }
    */

    /*
    csv.open("EventType.csv", ios::app|ios::out);
    if (csv.bad())
    {
      G4cout<<" Csv failed"<<G4endl;
    }
    else
    {
      G4cout << "... write Csv File : EventType.csv ";
      for(int k=0; k<8; k++)
      {
        csv << fRun->GetEventType(k);
        if(k<7)
        {
          csv << ",";
        }
        else
        {
          csv << endl;
        }
      }
      csv.close();
      G4cout << "- done." << G4endl;
    }
    */

    ///*
    csv.open("List.csv", ios::app|ios::out);
    if (!csv.is_open())
    {
      G4cout<<" Csv failed"<<G4endl;
    }
    else
    {
      G4cout << "... write Csv File : List.csv ";
      for(int i=0; i<fRun->GetTotalCount(); i++)
      {
          for(int j=0; j<5; j++)
          {
            csv << fRun->GetList(i, j);
            if(j<4)
            {
              csv << ",";
            }
          }
          
          if(i<(fRun->GetTotalCount()-1))
          {
            csv << "\n";
          }
          else
          {
            csv << endl;
          }
      }
      csv.close();
      G4cout << "- done, accepted events = " << fRun->GetTotalCount() << "." << G4endl;
    }
    //*/

  // show Rndm status
  G4Random::showEngineStatus();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
