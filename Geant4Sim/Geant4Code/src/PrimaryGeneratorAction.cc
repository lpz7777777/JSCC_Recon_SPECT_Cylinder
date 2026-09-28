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
    fCenterY = new G4UIcmdWithADouble("/xcat/centerY", this);
    fCenterY->SetGuidance("World Y coordinate of the source center in mm.");
    fEllipseDirectory = new G4UIdirectory("/ellipse/");
    fEllipseClear = new G4UIcmdWithoutParameter("/ellipse/clear", this);
    fEllipseAdd = new G4UIcmdWithAString("/ellipse/add", this);
    fEllipseAdd->SetGuidance("Add: energy_keV semiX semiY halfZ intensity (mm, local FOV coordinates).");
    fEllipseRod = new G4UIcmdWithAString("/ellipse/addRod", this);
    fEllipseRod->SetGuidance("Add: energy_keV x y z radius halfZ intensity (mm, local FOV coordinates).");
    fEllipseAngle = new G4UIcmdWithADouble("/ellipse/angle", this);
    fEllipseCenterY = new G4UIcmdWithADouble("/ellipse/centerY", this);
  }
  ~XcatSourceMessenger() override {
    delete fEllipseCenterY;
    delete fEllipseAngle;
    delete fEllipseAdd;
    delete fEllipseRod;
    delete fEllipseClear;
    delete fEllipseDirectory;
    delete fCenterY;
    delete fAngle;
    delete fAdd;
    delete fClear;
    delete fDirectory;
  }
  void SetNewValue(G4UIcommand* command, G4String value) override {
    if (command == fClear) fOwner->ClearXcatSources();
    else if (command == fAdd) fOwner->AddXcatSource(value);
    else if (command == fAngle) fOwner->SetXcatAngle(fAngle->GetNewDoubleValue(value));
    else if (command == fCenterY) fOwner->SetSourceCenterY(fCenterY->GetNewDoubleValue(value));
    else if (command == fEllipseClear) fOwner->ClearEllipseSources();
    else if (command == fEllipseAdd) fOwner->AddEllipseSource(value);
    else if (command == fEllipseRod) fOwner->AddEllipseRod(value);
    else if (command == fEllipseAngle) fOwner->SetXcatAngle(fEllipseAngle->GetNewDoubleValue(value));
    else if (command == fEllipseCenterY) fOwner->SetSourceCenterY(fEllipseCenterY->GetNewDoubleValue(value));
  }
 private:
  PrimaryGeneratorAction* fOwner;
  G4UIdirectory* fDirectory;
  G4UIcmdWithoutParameter* fClear;
  G4UIcmdWithAString* fAdd;
  G4UIcmdWithADouble* fAngle;
  G4UIcmdWithADouble* fCenterY;
  G4UIdirectory* fEllipseDirectory;
  G4UIcmdWithoutParameter* fEllipseClear;
  G4UIcmdWithAString* fEllipseAdd;
  G4UIcmdWithAString* fEllipseRod;
  G4UIcmdWithADouble* fEllipseAngle;
  G4UIcmdWithADouble* fEllipseCenterY;
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

void PrimaryGeneratorAction::SetSourceCenterY(G4double centerYmm)
{
  if (!std::isfinite(centerYmm))
    G4Exception("PrimaryGeneratorAction::SetSourceCenterY", "SRC001",
                FatalException, "Source center must be finite.");
  fSourceCenterY = centerYmm;
}

void PrimaryGeneratorAction::ClearEllipseSources()
{
  fEllipseSources.clear();
  fEllipseCumulative.clear();
  fEllipseTotal = 0;
  fUseEllipse = false;
}

void PrimaryGeneratorAction::AddEllipseSource(const G4String& specification)
{
  EllipseSource source{};
  std::istringstream input(specification);
  input >> source.energyKeV >> source.semiX >> source.semiY
        >> source.halfZ >> source.intensity;
  std::string extra;
  if (!input || (source.energyKeV != 218 && source.energyKeV != 440) ||
      !std::isfinite(source.semiX) || !std::isfinite(source.semiY) ||
      !std::isfinite(source.halfZ) || !std::isfinite(source.intensity) ||
      source.semiX <= 0 || source.semiY <= 0 || source.halfZ <= 0 ||
      source.intensity <= 0 || (input >> extra)) {
    G4Exception("PrimaryGeneratorAction::AddEllipseSource", "ELLIPSE001",
                FatalException, "Invalid /ellipse/add source specification.");
  }
  if (fUseXcat)
    G4Exception("PrimaryGeneratorAction::AddEllipseSource", "ELLIPSE002",
                FatalException, "XCAT and ellipse sources cannot be mixed.");
  fEllipseSources.push_back(source);
  fEllipseCumulative.clear();
  fUseEllipse = true;
}

void PrimaryGeneratorAction::AddEllipseRod(const G4String& specification)
{
  EllipseSource source{};
  std::istringstream input(specification);
  input >> source.energyKeV >> source.x >> source.y >> source.z
        >> source.semiX >> source.halfZ >> source.intensity;
  std::string extra;
  if (!input || (source.energyKeV != 218 && source.energyKeV != 440) ||
      !std::isfinite(source.x) || !std::isfinite(source.y) ||
      !std::isfinite(source.z) || !std::isfinite(source.semiX) ||
      !std::isfinite(source.halfZ) || !std::isfinite(source.intensity) ||
      source.semiX <= 0 || source.halfZ <= 0 || source.intensity <= 0 ||
      (input >> extra)) {
    G4Exception("PrimaryGeneratorAction::AddEllipseRod", "ELLIPSE003",
                FatalException, "Invalid /ellipse/addRod source specification.");
  }
  source.isRod = true;
  if (fUseXcat)
    G4Exception("PrimaryGeneratorAction::AddEllipseRod", "ELLIPSE004",
                FatalException, "XCAT and ellipse sources cannot be mixed.");
  fEllipseSources.push_back(source);
  fEllipseCumulative.clear();
  fUseEllipse = true;
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
  if (fUseEllipse)
    G4Exception("PrimaryGeneratorAction::AddXcatSource", "XCAT003",
                FatalException, "XCAT and ellipse sources cannot be mixed.");
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
			(fSourceCenterY + localY * fXcatCos - localX * fXcatSin) * mm,
			localZ * mm));
		const auto cosTheta = 2 * G4UniformRand() - 1;
		const auto phi = twopi * G4UniformRand();
		const auto sinTheta = std::sqrt(1 - cosTheta * cosTheta);
		fXcatGun->SetParticleMomentumDirection(G4ThreeVector(
			sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta));
		fXcatGun->SetParticleEnergy(box.energyKeV * keV);
		fXcatGun->GeneratePrimaryVertex(anEvent);
	} else if (fUseEllipse) {
		if (fEllipseCumulative.empty()) {
			fEllipseTotal = 0;
			for (const auto& source : fEllipseSources) {
				fEllipseTotal += source.intensity;
				fEllipseCumulative.push_back(fEllipseTotal);
			}
		}
		const auto draw = G4UniformRand() * fEllipseTotal;
		const auto selected = std::lower_bound(fEllipseCumulative.begin(),
		                                       fEllipseCumulative.end(), draw);
		const auto index = std::min(static_cast<std::size_t>(selected - fEllipseCumulative.begin()),
		                            fEllipseSources.size() - 1);
		const auto& source = fEllipseSources[index];
		const auto radial = std::sqrt(G4UniformRand());
		const auto polar = twopi * G4UniformRand();
		const auto localX = source.x + source.semiX * radial * std::cos(polar);
		const auto localY = source.y + (source.isRod ? source.semiX : source.semiY) * radial * std::sin(polar);
		const auto localZ = source.z + (2 * G4UniformRand() - 1) * source.halfZ;
		fXcatGun->SetParticlePosition(G4ThreeVector(
			(localX * fXcatCos + localY * fXcatSin) * mm,
			(fSourceCenterY + localY * fXcatCos - localX * fXcatSin) * mm,
			localZ * mm));
		const auto cosTheta = 2 * G4UniformRand() - 1;
		const auto phi = twopi * G4UniformRand();
		const auto sinTheta = std::sqrt(1 - cosTheta * cosTheta);
		fXcatGun->SetParticleMomentumDirection(G4ThreeVector(
			sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta));
		fXcatGun->SetParticleEnergy(source.energyKeV * keV);
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

