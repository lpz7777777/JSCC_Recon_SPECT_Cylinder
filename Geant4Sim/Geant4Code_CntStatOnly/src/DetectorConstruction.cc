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
/// \file hadronic/Hadr03/src/DetectorConstruction.cc
/// \brief Implementation of the DetectorConstruction class
//
// $Id: DetectorConstruction.cc 70755 2013-06-05 12:17:48Z ihrivnac $
//

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"

#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SubtractionSolid.hh"


#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"
#include "G4RunManager.hh"
#include "G4Exception.hh"

#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction():
    G4VUserDetectorConstruction(),
    fScinMaterial(0), fWMaterial(0), fPSMaterial(0), fAirMaterial(0), fDetectorMessenger(0), scinLV(0), psLV(0), AirLV(0), WLV(0), nScinNum(0)
{
    ScinSize = 3 * mm;
    ScinHeight = 3 * mm;

    nx = 32;
    ny = 64;
    nlayer = 31;

    GapXY = 4.2 * mm;
    GapZ = 3 * mm;

    ShieldOuterSize = 90 * mm;
    ShieldInnerSize = 84 * mm;
    ShieldHeight = 111 * mm;
    ShieldX = 0;
    ShieldY = 0;
    ShieldZ = 0 * mm;

    DefineMaterials();
    SetScinMaterial("GAGG");
    SetPSMaterial("K9");
    SetWMaterial("W");

    const int crystalMatrixSize = nx * ny * nlayer;
    const int frontMatrixSize = nx * ny * (nlayer - 1);
    CrystalMatrix = new int[crystalMatrixSize]();
    ifstream file("CrystalMatrix.txt");
    if (!file.is_open())
    {
        G4Exception("DetectorConstruction::DetectorConstruction",
            "Detector001", FatalException, "Cannot open CrystalMatrix.txt.");
    }

    for(int i=0; i<crystalMatrixSize; i++)
    {
        if (!(file >> CrystalMatrix[i]))
        {
            G4ExceptionDescription message;
            message << "CrystalMatrix.txt ended after " << i
                << " values; expected " << crystalMatrixSize << ".";
            G4Exception("DetectorConstruction::DetectorConstruction",
                "Detector002", FatalException, message);
        }
        if (CrystalMatrix[i] < 0 || CrystalMatrix[i] > 2)
        {
            G4ExceptionDescription message;
            message << "Unsupported CrystalMatrix label " << CrystalMatrix[i]
                << " at zero-based index " << i << ".";
            G4Exception("DetectorConstruction::DetectorConstruction",
                "Detector003", FatalException, message);
        }
    }
    int extraValue = 0;
    if (file >> extraValue)
    {
        G4Exception("DetectorConstruction::DetectorConstruction",
            "Detector004", FatalException,
            "CrystalMatrix.txt contains more than 32x64x31 values.");
    }
    file.close();

    int frontScinNum = 0;
    for(int i=0; i<frontMatrixSize; i++)
    {
        if (CrystalMatrix[i] == 1) frontScinNum++;
    }
    const int refinedLastLayerScinNum = 2 * ny * 2 * nx;
    nScinNum = frontScinNum + refinedLastLayerScinNum;
    if (frontScinNum != 2304 || nScinNum != 10496)
    {
        G4ExceptionDescription message;
        message << "Detector geometry mismatch: front 30 layers contain "
            << frontScinNum << " scintillators and the refined last layer contains "
            << refinedLastLayerScinNum << "; expected 2304 + 8192 = 10496.";
        G4Exception("DetectorConstruction::DetectorConstruction",
            "Detector005", FatalException, message);
    }
    G4cout << "Loaded CrystalMatrix.txt: 32x64x31; using 2304 scintillators "
        << "from layers 1-30 and replacing layer 31 with 8192 fine crystals."
        << G4endl;

    fDetectorMessenger = new DetectorMessenger(this);
    putmethod = new int*[nx];
    putmethod_Pinhole = new int*[nx];
    for(int i=0; i<nx; i++)
    {
        putmethod[i]=new int[ny];
        putmethod_Pinhole[i]=new int[ny];
    }

    //Initialization
    bool Isset=true;
    if(Isset){
        int putmethod1[28][28] =
        {
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        };

        int putmethod2[28][28] =
        {
            2,0,0,0,2,0,2,2,0,0,0,0,2,2,2,2,0,0,0,0,2,2,0,2,0,0,0,2,
            2,0,2,2,2,2,0,0,0,2,2,0,0,0,0,0,0,2,2,0,0,0,2,2,2,2,0,2,
            0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,
            0,2,0,0,2,2,0,0,0,2,2,0,0,0,0,0,0,2,2,0,0,0,2,2,0,0,2,0,
            0,0,0,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,0,0,0,
            2,0,0,2,0,0,2,0,0,2,2,2,0,0,0,0,2,2,2,0,0,2,0,0,2,0,0,2,
            0,2,0,0,0,0,0,0,0,2,0,2,0,2,2,0,2,0,2,0,0,0,0,0,0,0,2,0,
            2,0,0,2,0,2,0,2,2,0,0,0,0,0,0,0,0,0,0,2,2,0,2,0,2,0,0,2,
            2,0,2,0,0,2,2,0,0,0,0,0,0,2,2,0,0,0,0,0,0,2,2,0,0,2,0,2,
            2,0,2,0,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,2,0,2,
            2,0,0,0,2,0,0,0,0,0,2,0,0,2,2,0,0,2,0,0,0,0,0,2,0,0,0,2,
            0,2,0,0,2,0,2,2,0,0,2,0,0,0,0,0,0,2,0,0,2,2,0,2,0,0,2,0,
            2,0,0,0,2,0,2,0,2,2,0,0,2,0,0,2,0,0,2,2,0,2,0,2,0,0,0,2,
            0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,
            2,0,0,0,2,0,2,0,2,2,0,0,2,0,0,2,0,0,2,2,0,2,0,2,0,0,0,2,
            0,2,0,0,2,0,2,2,0,0,2,0,0,0,0,0,0,2,0,0,2,2,0,2,0,0,2,0,
            2,0,0,0,2,0,0,0,0,0,2,0,0,2,2,0,0,2,0,0,0,0,0,2,0,0,0,2,
            2,0,2,0,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,2,0,2,
            2,0,2,0,0,2,2,0,0,0,0,0,0,2,2,0,0,0,0,0,0,2,2,0,0,2,0,2,
            2,0,0,2,0,2,0,2,2,0,0,0,0,0,0,0,0,0,0,2,2,0,2,0,2,0,0,2,
            0,2,0,0,0,0,0,0,0,2,0,2,0,2,2,0,2,0,2,0,0,0,0,0,0,0,2,0,
            2,0,0,2,0,0,2,0,0,2,2,2,0,0,0,0,2,2,2,0,0,2,0,0,2,0,0,2,
            0,0,0,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,0,0,0,
            0,2,0,0,2,2,0,0,0,2,2,0,0,0,0,0,0,2,2,0,0,0,2,2,0,0,2,0,
            0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,
            2,0,2,2,2,2,0,0,0,2,2,0,0,0,0,0,0,2,2,0,0,0,2,2,2,2,0,2,
            2,0,0,0,2,0,2,2,0,0,0,0,2,2,2,2,0,0,0,0,2,2,0,2,0,0,0,2
        };

        /*
        for(int i=0; i<nx; i++)
        {
            for(int j=0; j<ny; j++)
            {
                G4cout << i << "," << j << G4endl;
                putmethod[i][j] = putmethod1[i][j];
                putmethod_Pinhole[i][j] = putmethod2[i][j];
            }
        }
        */

    }
    else
    {
        for(int i=1; i<=nx; i++)
        {
            for(int j=1; j<=ny; j++)
            {
                if(i%2==j%2)
                {
                    putmethod[i-1][j-1] = 1;
                }
                else
                {
                    putmethod[i-1][j-1] = 0;
                }
            }
        }
    }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{
    delete fDetectorMessenger;
    for(int i=0; i<nx; i++)
    {
        delete[] putmethod[i];
        delete[] putmethod_Pinhole[i];
    }
    delete[] putmethod;
    delete[] putmethod_Pinhole;
    delete[] CrystalMatrix;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    return ConstructVolumes();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::DefineMaterials()
{
    G4double a,z,density;
    G4String name, symbol;
    G4int natoms, ncomponents;
    G4double fractionmass;


    //----Define Element----
    G4NistManager* man = G4NistManager::Instance();

    // Aluminium
    G4Element* elAl = man->FindOrBuildElement("Al");
    // Gallium
    G4Element* elGa = man->FindOrBuildElement("Ga");
    // Gadolinium
    G4Element* elGd = man->FindOrBuildElement("Gd");
    // Cerium
    G4Element* elCe = man->FindOrBuildElement("Ce");
    // Boron
    G4Element* elB = man->FindOrBuildElement("B");
    // Barium
    G4Element* elBa = man->FindOrBuildElement("Ba");
    // Oxygen
    G4Element* elO  = man->FindOrBuildElement("O");
    // Silicon
    G4Element* elSi  = man->FindOrBuildElement("Si");
    // Natrium
    G4Element* elNa  = man->FindOrBuildElement("Na");
    // Kalium
    G4Element* elK  = man->FindOrBuildElement("K");
    // Arsenic
    G4Element* elAs  = man->FindOrBuildElement("As");
    // W
    G4Element* elW  = man->FindOrBuildElement("W");
    // Lu
    G4Element* elLu  = man->FindOrBuildElement("Lu");

    //----Define Material----
    // Air
    G4Material* Air = man->FindOrBuildMaterial("G4_AIR");

    // Vacuum
    G4Material* Vacuum = new G4Material("Vacuum",0.000000000000001*g/cm3, Air);

    // GAGG--(Gd3Al2Ga3O12)
    density = 6.6 * g/cm3;
    G4Material* GAGG1 = new G4Material(name="GAGG1", density, ncomponents=4);
    GAGG1->AddElement(elGd, natoms=3);
    GAGG1->AddElement(elAl, natoms=2);
    GAGG1->AddElement(elGa, natoms=3);
    GAGG1->AddElement(elO,  natoms=12);

    // GAGG:Ce Scintillators (Ce is 1%)
    density = 6.6 * g/cm3;
    G4Material* GAGG = new G4Material(name="GAGG", density, ncomponents=2);
    GAGG->AddMaterial(GAGG1, fractionmass=99*perCent);
    GAGG->AddElement(elCe, fractionmass=1*perCent);

    // Lu2SiO5
    density = 7.3 * g/cm3;
    G4Material* LSO = new G4Material(name="LSO", density, ncomponents=3);
    LSO->AddElement(elLu, natoms=2);
    LSO->AddElement(elSi, natoms=1);
    LSO->AddElement(elO, natoms=5);

    // SiO2
    density = 2.2 * g/cm3;
    G4Material* SiO2 = new G4Material(name="SiO2", density, ncomponents=2);
    SiO2->AddElement(elSi, natoms=1);
    SiO2->AddElement(elO, natoms=2);

    // B2O3
    density = 2.46 * g/cm3;
    G4Material* B2O3 = new G4Material(name="B2O3", density, ncomponents=2);
    B2O3->AddElement(elB, natoms=2);
    B2O3->AddElement(elO, natoms=3);

    // BaO
    density = 5.72 * g/cm3;
    G4Material* BaO = new G4Material(name="BaO", density, ncomponents=2);
    BaO->AddElement(elBa, natoms=1);
    BaO->AddElement(elO, natoms=1);

    // Na2O
    density = 2.27 * g/cm3;
    G4Material* Na2O = new G4Material(name="Na2O", density, ncomponents=2);
    Na2O->AddElement(elNa, natoms=2);
    Na2O->AddElement(elO, natoms=1);

    // K2O
    density = 2.3 * g/cm3;
    G4Material* K2O = new G4Material(name="K2O", density, ncomponents=2);
    K2O->AddElement(elK, natoms=2);
    K2O->AddElement(elO, natoms=1);

    // As2O3
    density = 3.74 * g/cm3;
    G4Material* As2O3 = new G4Material(name="As2O3", density, ncomponents=2);
    As2O3->AddElement(elAs, natoms=2);
    As2O3->AddElement(elO, natoms=3);

    // K9
    density = 2.51 * g/cm3;
    G4Material* K9 = new G4Material(name="K9", density, ncomponents=6);
    K9->AddMaterial(SiO2, fractionmass=69.13*perCent);
    K9->AddMaterial(B2O3, fractionmass=10.75*perCent);
    K9->AddMaterial(BaO, fractionmass=3.07*perCent);
    K9->AddMaterial(Na2O, fractionmass=10.40*perCent);
    K9->AddMaterial(K2O, fractionmass=6.29*perCent);
    K9->AddMaterial(As2O3, fractionmass=0.36*perCent);

    // W
    density = 19.35 * g/cm3;
    G4Material* W = new G4Material(name="W", density, ncomponents=1);
    W->AddElement(elW, natoms=1);


    // Print
    G4cout << *(G4Material::GetMaterialTable()) << G4endl;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::ConstructVolumes()
{
    // Cleanup old geometry
    G4GeometryManager::GetInstance()->OpenGeometry();
    G4PhysicalVolumeStore::GetInstance()->Clean();
    G4LogicalVolumeStore::GetInstance()->Clean();
    G4SolidStore::GetInstance()->Clean();

    // Air Material
    G4NistManager* man = G4NistManager::Instance();
    G4Material* Vac = man->FindOrBuildMaterial("Vacuum");

    // World box
    G4Box* worldSBox = new G4Box("World", 1*m, 1*m, 1*m);
    G4LogicalVolume* worldLBox =
    new G4LogicalVolume(    worldSBox,                //its shape
                            Vac,                    //its material
                            "lWorld");                //its name
    G4VPhysicalVolume* worldPBox =
    new G4PVPlacement(        0,                        //no rotation
                            G4ThreeVector(),        //at(0,0,0)
                            worldLBox,                //its logical volume
                            "pWorld",                //its name
                            0,                        //world
                            false,                    //no boolean operation
                            0);                        //copy number

    // Scintillator
    // Define a scintillator box
    G4Box* ScinBox = new G4Box("ScinBox", ScinSize/2., ScinSize/2., ScinHeight/2.);
    scinLV = new G4LogicalVolume(ScinBox, fScinMaterial, "ScinLV", 0, 0, 0);

    G4Box* ScinBox2 = new G4Box("ScinBox2", 1*mm, 3*mm, 1*mm);
    scinLV2 = new G4LogicalVolume(ScinBox2, fScinMaterial, "ScinLV2", 0, 0, 0);

    // Print
    G4cout << "Define a scintillator box -- Done" << G4endl;

    // Define WBox
    G4Box* WBox = new G4Box("WBox", ScinSize/2., ScinSize/2., ScinHeight/2.);
    WLV = new G4LogicalVolume(WBox, fWMaterial, "WLV", 0, 0, 0);
    G4cout << "Define a W box -- Done" << G4endl;

    // Define Shield
    /*
    G4Box* WBox1 = new G4Box("WBox1", (ShieldOuterSize)/2., ShieldOuterSize/2., ShieldHeight/2.);
    G4Box* WBox2 = new G4Box("WBox1", (ShieldInnerSize)/2., ShieldInnerSize/2., (ShieldHeight+1)/2.);
    G4SubtractionSolid* ShieldBox = new G4SubtractionSolid("ShieldBox", WBox1, WBox2, 0, G4ThreeVector(0, 0, 0));
    ShieldLV = new G4LogicalVolume(ShieldBox, fWMaterial, "ShieldLV", 0, 0, 0);
    new G4PVPlacement(    0,
                G4ThreeVector(ShieldX, ShieldY, ShieldZ),
                ShieldLV,
                "Shield",
                worldLBox,
                false,
                0,
                true);
    // Print
    G4cout << "Set Shield -- Done" << G4endl;
    */

    // Define Phantom
    G4Material* Air = man->FindOrBuildMaterial("G4_AIR");
    G4Material* Vacuum = new G4Material("Vacuum",0.000000000000001*g/cm3, Air);
    // G4Tubs* ContrastPhantomTub = new G4Tubs("ContrastPhantomTub", 0, 140*mm, 0.01*mm, 0*deg, 360*deg);
    // G4Tubs* HoleTub1 = new G4Tubs("ContrastPhantomTub", 0, 15*mm, 0.02*mm, 0*deg, 360*deg);
    // G4Tubs* HoleTub2 = new G4Tubs("ContrastPhantomTub", 0, 18*mm, 0.02*mm, 0*deg, 360*deg);
    // G4Tubs* HoleTub3 = new G4Tubs("ContrastPhantomTub", 0, 21*mm, 0.02*mm, 0*deg, 360*deg);
    // G4Tubs* HoleTub4 = new G4Tubs("ContrastPhantomTub", 0, 24*mm, 0.02*mm, 0*deg, 360*deg);
    // G4Tubs* HoleTub5 = new G4Tubs("ContrastPhantomTub", 0, 27*mm, 0.02*mm, 0*deg, 360*deg);
    // G4Tubs* HoleTub6 = new G4Tubs("ContrastPhantomTub", 0, 30*mm, 0.02*mm, 0*deg, 360*deg);
    // G4SubtractionSolid* ContrastPhantomBox = new G4SubtractionSolid("ContrastPhantomBox", ContrastPhantomTub, HoleTub1, 0, G4ThreeVector(2*35*mm, 0*mm, 0));
    // ContrastPhantomBox = new G4SubtractionSolid("ContrastPhantomBox", ContrastPhantomBox, HoleTub2, 0, G4ThreeVector(2*17.5*mm, 2*30.3109*mm, 0));
    // ContrastPhantomBox = new G4SubtractionSolid("ContrastPhantomBox", ContrastPhantomBox, HoleTub3, 0, G4ThreeVector(-2*17.5*mm, 2*30.3109*mm, 0));
    // ContrastPhantomBox = new G4SubtractionSolid("ContrastPhantomBox", ContrastPhantomBox, HoleTub4, 0, G4ThreeVector(-2*35*mm, 0*mm, 0));
    // ContrastPhantomBox = new G4SubtractionSolid("ContrastPhantomBox", ContrastPhantomBox, HoleTub5, 0, G4ThreeVector(-2*17.5*mm, -2*30.3109*mm, 0));
    // ContrastPhantomBox = new G4SubtractionSolid("ContrastPhantomBox", ContrastPhantomBox, HoleTub6, 0, G4ThreeVector(2*17.5*mm, -2*30.3109*mm, 0));

    // G4cout << "Define the ContrastPhantomBox -- Done" << G4endl;
    // G4LogicalVolume* ContrastPhantomLV = new G4LogicalVolume(ContrastPhantomBox, Vacuum, "ContrastPhantomLV", 0, 0, 0);
    // G4cout << "Define the G4LogicalVolume -- Done" << G4endl;
    // new G4PVPlacement(0, G4ThreeVector(0*mm, 0*mm, -146.5*mm), ContrastPhantomLV, "ContrastPhantom", worldLBox, false, 0, true);
    // G4cout << "Define the collimator -- Done" << G4endl;


    // Set Scintillator
    ///*
    int OutputDet = 0;

    ofstream csv;
    if(OutputDet)
    {
        csv.open("Detector.csv",ios::app|ios::out);
    }

    G4double tempx, tempy, tempz;

    int tempn = 1;
    int IdCrystal = 0;

    for(int layer=1; layer<nlayer; layer++)
    {
        tempy = GapZ * ((G4double)layer - ((G4double)(nlayer + 1) / 2.));
        for(int j=1; j<=ny; j++)
        {
            tempx = GapXY * (- ny / 2. + j - 1.0 / 2);

            for(int i=1; i<=nx; i++)
            {
                tempz = GapXY * (- nx / 2. + i - 1.0 / 2);
                if (CrystalMatrix[IdCrystal] == 1)
                {
                    new G4PVPlacement(    0,
                        G4ThreeVector(tempx, tempy, tempz),
                        scinLV,
                        "Scin",
                        worldLBox,
                        false,
                        tempn,
                        true);

                    if(OutputDet)
                    {
                        csv << tempn << "," << tempx << "," << tempy << "," << tempz << "\n";
                    }
                    tempn += 1;
                }

                else if(CrystalMatrix[IdCrystal] == 2)
                {
                    new G4PVPlacement(    0,
                                        G4ThreeVector(tempx, tempy, tempz),
                                        WLV,
                                        "W",
                                        worldLBox,
                                        false,
                                        0,
                                        true);
                }

                IdCrystal += 1;
            }
        }
    }

    ////
    tempy = GapZ * ((G4double)nlayer - ((G4double)(nlayer + 1) / 2.));
    for(int j=1; j<=2*ny; j++)
    {
        tempx = 2.1*mm * (- 2*ny / 2. + j - 1.0 / 2);
        for (int i=1; i<=2*nx; i++)
        {
            tempz = 2.1*mm * (- 2*nx / 2. + i - 1.0 / 2);
            new G4PVPlacement(    0,
                G4ThreeVector(tempx, tempy, tempz),
                scinLV2,
                "Scin",
                worldLBox,
                false,
                tempn,
                true);

            if(OutputDet)
            {
                csv << tempn << "," << tempx << "," << tempy << "," << tempz << "\n";
            }
            tempn += 1;
        }

    }

    if (tempn - 1 != nScinNum)
    {
        G4ExceptionDescription message;
        message << "Placed " << tempn - 1 << " scintillators, expected "
            << nScinNum << ".";
        G4Exception("DetectorConstruction::ConstructVolumes",
            "Detector006", FatalException, message);
    }
    G4cout << "Set Scin -- Done: " << nScinNum << " scintillators" << G4endl;


    // ////Attributes
    // G4VisAttributes* VisAtt;
    // //Detector Color
    // VisAtt = new G4VisAttributes(G4Colour::G4Colour(1.0, 1.0, 0.0, 1));
    // VisAtt->SetForceWireframe(true);
    // VisAtt->SetForceSolid(true);
    // VisAtt->SetLineWidth(3*mm);
    // scinLV->SetVisAttributes(VisAtt);
    // scinLV2->SetVisAttributes(VisAtt);
    // // Collimator Color
    // //World color
    // VisAtt = new G4VisAttributes(G4Colour::G4Colour(0.5, 0.5, 0.5, 1));
    // VisAtt->SetForceWireframe(true);
    // VisAtt->SetForceSolid(true);
    // VisAtt->SetLineWidth(3*mm);
    // WLV->SetVisAttributes(VisAtt);
    // worldLBox->SetVisAttributes(new G4VisAttributes(G4Colour::White()));


    //Print
    G4cout << "Constructing volumes success!" << G4endl;

    //always return the root volume
    return worldPBox;
}



//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::SetScinMaterial(G4String materialChoice)
{
      // search the material by its name
    G4Material* pttoMaterial = G4NistManager::Instance()->FindOrBuildMaterial(materialChoice);

      if (pttoMaterial)
    {
        if(fScinMaterial != pttoMaterial)
        {
              fScinMaterial = pttoMaterial;
             if(scinLV)
            {
                scinLV->SetMaterial(pttoMaterial);
            }
              G4RunManager::GetRunManager()->PhysicsHasBeenModified();
        }
      }
    else
    {
            G4cout << "\n--> warning from DetectorConstruction::SetScinMaterial : " << materialChoice << " not found" << G4endl;
      }
}

void DetectorConstruction::SetPSMaterial(G4String materialChoice)
{
    // search the material by its name
      G4Material* pttoMaterial = G4NistManager::Instance()->FindOrBuildMaterial(materialChoice);

      if (pttoMaterial)
    {
        if(fPSMaterial != pttoMaterial)
        {
              fPSMaterial = pttoMaterial;
              if(psLV)
            {
                psLV->SetMaterial(pttoMaterial);
            }
              G4RunManager::GetRunManager()->PhysicsHasBeenModified();
        }
      }
    else
    {
        G4cout << "\n--> warning from DetectorConstruction::SetPSMaterial : " << materialChoice << " not found" << G4endl;
      }
}

void DetectorConstruction::SetWMaterial(G4String materialChoice)
{
    // search the material by its name
      G4Material* pttoMaterial = G4NistManager::Instance()->FindOrBuildMaterial(materialChoice);

      if (pttoMaterial)
    {
        if(fWMaterial != pttoMaterial)
        {
              fWMaterial = pttoMaterial;
              if(WLV)
            {
                WLV->SetMaterial(pttoMaterial);
            }
              G4RunManager::GetRunManager()->PhysicsHasBeenModified();
        }
      }
    else
    {
        G4cout << "\n--> warning from DetectorConstruction::SetWMaterial : " << materialChoice << " not found" << G4endl;
      }
}

void DetectorConstruction::SetAirMaterial(G4String materialChoice)
{
      // search the material by its name
      G4Material* pttoMaterial = G4NistManager::Instance()->FindOrBuildMaterial(materialChoice);

      if (pttoMaterial)
    {
        if(fAirMaterial != pttoMaterial)
        {
              fAirMaterial = pttoMaterial;
              if(AirLV)
            {
                AirLV->SetMaterial(pttoMaterial);
            }
              G4RunManager::GetRunManager()->PhysicsHasBeenModified();
        }
      }
    else
    {
        G4cout << "\n--> warning from DetectorConstruction::SetAirMaterial : " << materialChoice << " not found" << G4endl;
      }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::SetScinSize(G4double value)
{
    ScinSize = value;
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetGapXY(G4double value)
{
    GapXY = value;
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetGapZ(G4double value)
{
    GapZ = value;
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetScinHeight(G4double value)
{
    ScinHeight = value;
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::Setnx(int value)
{
    nx = value;
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::Setny(int value)
{
    ny = value;
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}
