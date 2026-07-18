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
/// \file hadronic/Hadr03/include/DetectorConstruction.hh
/// \brief Definition of the DetectorConstruction class
//
// $Id: DetectorConstruction.hh 66586 2012-12-21 10:48:39Z ihrivnac $
//

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4LogicalVolume;
class G4Material;
class DetectorMessenger;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class DetectorConstruction : public G4VUserDetectorConstruction
{
      public:
        DetectorConstruction();
           ~DetectorConstruction();

      public:
        virtual G4VPhysicalVolume* Construct();

      public:
        // Set Size
          void Setnx(int value);
        void Setny(int value);
        void SetScinSize(G4double value);
        void SetScinHeight(G4double value);
        void SetGapXY(G4double value);
        void SetGapZ(G4double value);

        // Set Material
        void SetPSMaterial(G4String materialChoice);
        void SetScinMaterial(G4String materialChoice);
        void SetAirMaterial(G4String materialChoice);
        void SetWMaterial(G4String materialChoice);

        // Get Size
        int Getnx()const{return nx;}
        int Getny()const{return ny;}
        int Getlayers()const{return nlayer;}
        int GetnxCompton()const{return nx_Compton;}
        int GetnyCompton()const{return ny_Compton;}
        int GetlayersCompton()const{return nlayer_Compton;}
        int GetScinNum()const{return nScinNum;}
        int GetScinInfo(int numx,int numy){return putmethod[numx][numy];}
        G4double GetSize() const {return ScinSize;}
        G4double  GetLength() const {return ScinHeight;}

        // Get Material
        G4Material* GetScinMaterial(){return fScinMaterial;}
        G4Material* GetPSMaterial(){return fPSMaterial;}
        G4Material* GetAirMaterial(){return fAirMaterial;}
        G4Material* GetWMaterial(){return fWMaterial;}


      private:
        // Size
        int** putmethod;
        int** putmethod_Pinhole;
        int* CrystalMatrix;

        int nx, ny, nlayer, Wlayer;
        int nx_Compton, ny_Compton, nlayer_Compton, nx_Pinhole, ny_Pinhole;
        int nx_SubPixel, ny_SubPixel;
        int nScinNum;
        int IsWater;

        G4double ScinSize;
        G4double ScinHeight;
        G4double ScinSize_Compton;
        G4double ScinHeight_Compton;

        G4double WaterSize;
        G4double WaterHeight;
        G4double WaterX;
        G4double WaterY;
        G4double WaterZ;

        G4double CollimatorX;
        G4double CollimatorY;
        G4double CollimatorZ;

        G4double GapXY, GapXY_Compton;
        G4double GapZ, GapZ_Compton;
        G4double CollimatorSize, CollimatorHeight;
        G4double GapX_Pinhole, GapY_Pinhole, D_Pinhole;

        G4double ShieldOuterSize;
        G4double ShieldInnerSize;
        G4double ShieldHeight;
        G4double ShieldX;
        G4double ShieldY;
        G4double ShieldZ;

        // Material
        G4Material* fScinMaterial;
        G4Material* fPSMaterial;
        G4Material* fAirMaterial;
        G4Material* fWMaterial;

        // Logical Volume
        G4LogicalVolume* scinLV;
        G4LogicalVolume* scinLV_Compton;
        G4LogicalVolume* psLV;
        G4LogicalVolume* AirLV;
        G4LogicalVolume* WLV;
        G4LogicalVolume* scinLV2;
        G4LogicalVolume* ShieldLV;

        // Detector Messenge
        DetectorMessenger* fDetectorMessenger;

      private:
         void DefineMaterials();
         G4VPhysicalVolume* ConstructVolumes();
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


#endif
