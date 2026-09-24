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
/// \file electromagnetic/TestEm4/src/PrimaryGeneratorAction.cc
/// \brief Implementation of the PrimaryGeneratorAction class
//
//
// $Id: PrimaryGeneratorAction.cc 67268 2013-02-13 11:38:40Z ihrivnac $
//
// 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include <math.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <string>

#include "PrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4GeneralParticleSource.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "G4RunManager.hh"
#include "G4UImessenger.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4Exception.hh"
#include "DetectorConstruction.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

namespace {
class XcatSourceMessenger final : public G4UImessenger {
 public:
  explicit XcatSourceMessenger(PrimaryGeneratorAction* owner) : fOwner(owner) {
    fDirectory = new G4UIdirectory("/xcat/");
    fDirectory->SetGuidance("XCAT dual-energy voxel source commands.");
    fClear = new G4UIcmdWithoutParameter("/xcat/clear", this);
    fAdd = new G4UIcmdWithAString("/xcat/add", this);
    fAdd->SetGuidance("Add: energy_keV x y z hx hy hz intensity (mm, local FOV coordinates).");
    fAngle = new G4UIcmdWithADouble("/xcat/angle", this);
    fAngle->SetGuidance("Rotate the XCAT source by this angle in degrees.");
  }
  ~XcatSourceMessenger() override {
    delete fAngle;
    delete fAdd;
    delete fClear;
    delete fDirectory;
  }
  void SetNewValue(G4UIcommand* command, G4String value) override {
    if (command == fClear) fOwner->ClearXcatSources();
    else if (command == fAdd) fOwner->AddXcatSource(value);
    else if (command == fAngle) fOwner->SetXcatAngle(fAngle->GetNewDoubleValue(value));
  }
 private:
  PrimaryGeneratorAction* fOwner;
  G4UIdirectory* fDirectory;
  G4UIcmdWithoutParameter* fClear;
  G4UIcmdWithAString* fAdd;
  G4UIcmdWithADouble* fAngle;
};
}

PrimaryGeneratorAction::PrimaryGeneratorAction():
	G4VUserPrimaryGeneratorAction()
{
	// Define Parameter
	fParticleGun = new G4GeneralParticleSource();
	fXcatGun = new G4ParticleGun(1);
	fXcatMessenger = new XcatSourceMessenger(this);

	particleName = "gamma";
	// Batch macros replace this fallback and configure the full 218/440 keV
	// multi-source mixture. Keep a relevant default for interactive checks.
	energy = 218 * keV;
	position = G4ThreeVector(0*mm, 0*mm, -110*mm);
	// radiu = 1 * mm;
	
	// default particle kinematic
	G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
	G4ParticleDefinition* particle = particleTable->FindParticle(particleName);
	fParticleGun->SetParticleDefinition(particle);
	fXcatGun->SetParticleDefinition(particle);

	// DEFINE ENERGETIC DISTRIBUTION
	G4SPSEneDistribution *eneDist = fParticleGun->GetCurrentSource()->GetEneDist() ;
	eneDist->SetMonoEnergy(energy);

	// SET POSITION DISTRIBUTION 
	G4SPSPosDistribution *posDist = fParticleGun->GetCurrentSource()->GetPosDist() ;
	
	/*
	posDist->SetCentreCoords(position);
	posDist->SetPosDisType("Plane");
	posDist->SetPosDisShape("Square");
	posDist->SetHalfX(radiu);
	posDist->SetHalfY(radiu);
	*/
	posDist->SetPosDisType("Point");
	posDist->SetCentreCoords(position);	
	

	// SET ANGULAR DISTRIBUTION 
	G4SPSAngDistribution *angDist = fParticleGun->GetCurrentSource()->GetAngDist() ;
	angDist->SetAngDistType("iso");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fXcatMessenger;
    delete fXcatGun;
  	delete fParticleGun;
}

void PrimaryGeneratorAction::ClearXcatSources()
{
  fXcatBoxes.clear();
  fXcatCumulative.clear();
  fXcatTotal = 0;
  fUseXcat = false;
}

void PrimaryGeneratorAction::AddXcatSource(const G4String& specification)
{
  XcatBox box{};
  std::istringstream input(specification);
  input >> box.energyKeV >> box.x >> box.y >> box.z
        >> box.hx >> box.hy >> box.hz >> box.intensity;
  std::string extra;
  if (!input || (box.energyKeV != 218 && box.energyKeV != 440) ||
      !std::isfinite(box.x) || !std::isfinite(box.y) || !std::isfinite(box.z) ||
      !std::isfinite(box.hx) || !std::isfinite(box.hy) || !std::isfinite(box.hz) ||
      !std::isfinite(box.intensity) || box.hx <= 0 || box.hy <= 0 ||
      box.hz <= 0 || box.intensity <= 0 || (input >> extra)) {
    G4Exception("PrimaryGeneratorAction::AddXcatSource", "XCAT001", FatalException,
                "Invalid /xcat/add source specification.");
  }
  fXcatBoxes.push_back(box);
  fXcatCumulative.clear();
  fUseXcat = true;
}

void PrimaryGeneratorAction::SetXcatAngle(G4double angleDegrees)
{
  fXcatCos = std::cos(angleDegrees * deg);
  fXcatSin = std::sin(angleDegrees * deg);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
	//this function is called at the beginning of event
	// GENERATION
	if (fUseXcat) {
		if (fXcatCumulative.empty()) {
			fXcatTotal = 0;
			fXcatCumulative.reserve(fXcatBoxes.size());
			for (const auto& box : fXcatBoxes) {
				fXcatTotal += box.intensity;
				fXcatCumulative.push_back(fXcatTotal);
			}
		}
		if (fXcatBoxes.empty()) G4Exception("PrimaryGeneratorAction::GeneratePrimaries",
		                              "XCAT002", FatalException, "No XCAT sources defined.");
		const auto draw = G4UniformRand() * fXcatTotal;
		const auto selected = std::lower_bound(fXcatCumulative.begin(), fXcatCumulative.end(), draw);
		const auto index = std::min(static_cast<std::size_t>(selected - fXcatCumulative.begin()),
		                            fXcatBoxes.size() - 1);
		const auto& box = fXcatBoxes[index];
		const auto localX = box.x + (2 * G4UniformRand() - 1) * box.hx;
		const auto localY = box.y + (2 * G4UniformRand() - 1) * box.hy;
		const auto localZ = box.z + (2 * G4UniformRand() - 1) * box.hz;
		fXcatGun->SetParticlePosition(G4ThreeVector(
			(localX * fXcatCos + localY * fXcatSin) * mm,
			(-245 + localY * fXcatCos - localX * fXcatSin) * mm,
			localZ * mm));
		const auto cosTheta = 2 * G4UniformRand() - 1;
		const auto phi = twopi * G4UniformRand();
		const auto sinTheta = std::sqrt(1 - cosTheta * cosTheta);
		fXcatGun->SetParticleMomentumDirection(G4ThreeVector(
			sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta));
		fXcatGun->SetParticleEnergy(box.energyKeV * keV);
		fXcatGun->GeneratePrimaryVertex(anEvent);
	} else {
		fParticleGun->GeneratePrimaryVertex(anEvent);
	}
	for (auto* vertex = anEvent->GetPrimaryVertex(); vertex; vertex = vertex->GetNext())
	{
		for (auto* primary = vertex->GetPrimary(); primary; primary = primary->GetNext())
		{
			const auto energyKeV = primary->GetKineticEnergy() / keV;
			if (std::abs(energyKeV - 218.0) < 0.001) ++fPrimary218;
			else if (std::abs(energyKeV - 440.0) < 0.001) ++fPrimary440;
			else ++fPrimaryOther;
		}
	}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

