#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Exception.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MultiUnion.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4RotationMatrix.hh"
#include "G4SubtractionSolid.hh"
#include "G4SystemOfUnits.hh"
#include "G4Transform3D.hh"
#include "G4Tubs.hh"
#include "G4VisAttributes.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace
{
constexpr G4double kToleranceMm = 1.0e-6;

void Require(G4bool condition, const char* code, const char* message)
{
  if (!condition)
  {
    G4Exception("DetectorConstruction", code, FatalException, message);
  }
}
}

DetectorConstruction::DetectorConstruction()
  : G4VUserDetectorConstruction(),
    fVacuum(nullptr),
    fLead(nullptr),
    fNaI(nullptr),
    fScintillatorLV(nullptr)
{
  DefineMaterials();
}

void DetectorConstruction::DefineMaterials()
{
  auto* nist = G4NistManager::Instance();
  fVacuum = nist->FindOrBuildMaterial("G4_Galactic");
  fLead = nist->FindOrBuildMaterial("G4_Pb");
  fNaI = nist->FindOrBuildMaterial("G4_SODIUM_IODIDE");

  Require(fVacuum != nullptr, "EHE001", "Cannot construct G4_Galactic.");
  Require(fLead != nullptr, "EHE002", "Cannot construct G4_Pb.");
  Require(fNaI != nullptr, "EHE003", "Cannot construct G4_SODIUM_IODIDE.");
}

std::vector<DetectorConstruction::HoleCenter> DetectorConstruction::BuildHoleCenters() const
{
  const G4double columnPitch = std::sqrt(3.0) * kHolePitch / 2.0;
  std::vector<HoleCenter> holes;
  holes.reserve(kHoleCount);

  // This is a direct translation of build_collimator.m. Its temporary px
  // coordinate becomes detector Z, while py becomes detector X.
  for (G4int row = 1; row <= kHoleRows; ++row)
  {
    const G4double offset = (row % 2 == 0) ? kHolePitch / 2.0 : 0.0;
    for (G4int column = 1; column <= kHoleColumns; ++column)
    {
      const G4double px = (row - 0.5 - kHoleRows / 2.0) * columnPitch;
      const G4double py = (column - 0.5 - kHoleColumns / 2.0) * kHolePitch + offset;
      holes.push_back({py, px});
    }
  }

  const G4double meanX = std::accumulate(
    holes.begin(), holes.end(), 0.0,
    [](G4double value, const HoleCenter& hole) { return value + hole.x; }) / holes.size();
  const G4double meanZ = std::accumulate(
    holes.begin(), holes.end(), 0.0,
    [](G4double value, const HoleCenter& hole) { return value + hole.z; }) / holes.size();
  for (auto& hole : holes)
  {
    hole.x -= meanX;
    hole.z -= meanZ;
  }
  return holes;
}

void DetectorConstruction::ValidateGeometry(const std::vector<HoleCenter>& holes) const
{
  Require(static_cast<G4int>(holes.size()) == kHoleCount,
          "EHE004", "Collimator hole count is not 1250.");

  G4double minimumDistance = std::numeric_limits<G4double>::max();
  for (std::size_t i = 0; i < holes.size(); ++i)
  {
    Require(std::abs(holes[i].x) + kHoleDiameter / 2.0 <= kCollimatorSizeX / 2.0 + kToleranceMm,
            "EHE005", "A hole exceeds the collimator X boundary.");
    Require(std::abs(holes[i].z) + kHoleDiameter / 2.0 <= kCollimatorSizeZ / 2.0 + kToleranceMm,
            "EHE006", "A hole exceeds the collimator Z boundary.");
    for (std::size_t j = i + 1; j < holes.size(); ++j)
    {
      const G4double dx = holes[i].x - holes[j].x;
      const G4double dz = holes[i].z - holes[j].z;
      minimumDistance = std::min(minimumDistance, std::sqrt(dx * dx + dz * dz));
    }
  }

  Require(std::abs(minimumDistance - kHolePitch) < kToleranceMm,
          "EHE007", "Nearest-neighbor hole pitch is not 5.9 mm.");
  Require(std::abs((minimumDistance - kHoleDiameter) - kSeptalThickness) < kToleranceMm,
          "EHE008", "Edge-to-edge septal thickness is not 3.4 mm.");

  const G4double collimatorBack = kCollimatorCenterY + kCollimatorThicknessY / 2.0;
  const G4double collimatorFront = kCollimatorCenterY - kCollimatorThicknessY / 2.0;
  const G4double detectorFront = kDetectorCenterY - kDetectorThicknessY / 2.0;
  Require(std::abs((collimatorFront - kFovCenterY) - kCommonFrontFaceDistance) < kToleranceMm,
          "EHE009", "Collimator front face is not 198.5 mm from the FOV center.");
  Require(std::abs(collimatorBack - detectorFront) < kToleranceMm,
          "EHE010", "NaI front face is not in contact with the collimator back face.");
  Require(kDetectorCount == 2312, "EHE011", "Detector count is not 2312.");
}

void DetectorConstruction::WriteGeometryFiles(const std::vector<HoleCenter>& holes) const
{
  std::ofstream holeFile("EHE_CollimatorHoles.csv", std::ios::out | std::ios::trunc);
  Require(holeFile.is_open(), "EHE012", "Cannot write EHE_CollimatorHoles.csv.");
  holeFile << "id,x_mm,y1_mm,y2_mm,z_mm,radius_mm\n";
  holeFile << std::setprecision(12);
  for (std::size_t i = 0; i < holes.size(); ++i)
  {
    holeFile << i + 1 << ',' << holes[i].x << ','
             << kCollimatorCenterY - kCollimatorThicknessY / 2.0 << ','
             << kCollimatorCenterY + kCollimatorThicknessY / 2.0 << ','
             << holes[i].z << ',' << kHoleDiameter / 2.0 << '\n';
  }

  std::ofstream detectorFile("EHE_DetectorGeometry.csv", std::ios::out | std::ios::trunc);
  Require(detectorFile.is_open(), "EHE013", "Cannot write EHE_DetectorGeometry.csv.");
  detectorFile << "id,x_mm,y_mm,z_mm,size_x_mm,size_y_mm,size_z_mm\n";
  detectorFile << std::setprecision(12);
  G4int copyNumber = 1;
  for (G4int xIndex = 1; xIndex <= kDetectorNx; ++xIndex)
  {
    const G4double x = kDetectorPixelX * (xIndex - kDetectorNx / 2.0 - 0.5);
    for (G4int zIndex = 1; zIndex <= kDetectorNz; ++zIndex)
    {
      const G4double z = kDetectorPixelZ * (zIndex - kDetectorNz / 2.0 - 0.5);
      detectorFile << copyNumber++ << ',' << x << ',' << kDetectorCenterY << ',' << z << ','
                   << kDetectorPixelX << ',' << kDetectorThicknessY << ',' << kDetectorPixelZ << '\n';
    }
  }

  std::ofstream summary("EHE_GeometrySummary.txt", std::ios::out | std::ios::trunc);
  Require(summary.is_open(), "EHE014", "Cannot write EHE_GeometrySummary.txt.");
  summary << std::setprecision(12)
          << "geometry = EHE triangular-lattice parallel-hole SPECT\n"
          << "fov_center_y_mm = " << kFovCenterY << '\n'
          << "common_front_face_distance_mm = " << kCommonFrontFaceDistance << '\n'
          << "fov_to_collimator_local_origin_mm = " << kFovToCollimatorOrigin << '\n'
          << "collimator_center_y_mm = " << kCollimatorCenterY << '\n'
          << "collimator_front_y_mm = "
          << kCollimatorCenterY - kCollimatorThicknessY / 2.0 << '\n'
          << "collimator_back_y_mm = "
          << kCollimatorCenterY + kCollimatorThicknessY / 2.0 << '\n'
          << "collimator_size_x_y_z_mm = " << kCollimatorSizeX << ','
          << kCollimatorThicknessY << ',' << kCollimatorSizeZ << '\n'
          << "hole_rows_columns_count = " << kHoleRows << ',' << kHoleColumns << ',' << kHoleCount << '\n'
          << "hole_diameter_pitch_septum_mm = " << kHoleDiameter << ',' << kHolePitch << ','
          << kSeptalThickness << '\n'
          << "detector_grid_x_z_count = " << kDetectorNx << ',' << kDetectorNz << ','
          << kDetectorCount << '\n'
          << "detector_pixel_x_y_z_mm = " << kDetectorPixelX << ',' << kDetectorThicknessY << ','
          << kDetectorPixelZ << '\n'
          << "detector_center_y_mm = " << kDetectorCenterY << '\n'
          << "collimator_material = G4_Pb\n"
          << "detector_material = G4_SODIUM_IODIDE\n";
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  const auto holes = BuildHoleCenters();
  ValidateGeometry(holes);
  WriteGeometryFiles(holes);

  auto* worldSolid = new G4Box("World", 1.0 * m, 1.0 * m, 1.0 * m);
  auto* worldLV = new G4LogicalVolume(worldSolid, fVacuum, "WorldLV");
  auto* worldPV = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "World", nullptr, false, 0, true);

  auto* plate = new G4Box("EHEPlateBox",
                          kCollimatorSizeX * mm / 2.0,
                          kCollimatorThicknessY * mm / 2.0,
                          kCollimatorSizeZ * mm / 2.0);
  auto* oneHole = new G4Tubs("EHEHoleCylinder", 0.0,
                             kHoleDiameter * mm / 2.0,
                             (kCollimatorThicknessY / 2.0 + 0.1) * mm,
                             0.0, 360.0 * deg);
  auto* allHoles = new G4MultiUnion("EHEHoleUnion");
  G4RotationMatrix holeRotation;
  holeRotation.rotateX(90.0 * deg);
  for (const auto& hole : holes)
  {
    allHoles->AddNode(*oneHole,
      G4Transform3D(holeRotation, G4ThreeVector(hole.x * mm, 0.0, hole.z * mm)));
  }
  allHoles->Voxelize();

  auto* collimatorSolid = new G4SubtractionSolid("EHECollimatorSolid", plate, allHoles);
  auto* collimatorLV = new G4LogicalVolume(collimatorSolid, fLead, "EHECollimatorLV");
  new G4PVPlacement(nullptr,
                    G4ThreeVector(0.0, kCollimatorCenterY * mm, 0.0),
                    collimatorLV, "EHECollimator", worldLV, false, 0, true);

  auto* detectorSolid = new G4Box("NaIPixelSolid",
                                  kDetectorPixelX * mm / 2.0,
                                  kDetectorThicknessY * mm / 2.0,
                                  kDetectorPixelZ * mm / 2.0);
  fScintillatorLV = new G4LogicalVolume(detectorSolid, fNaI, "ScinLV");

  G4int copyNumber = 1;
  for (G4int xIndex = 1; xIndex <= kDetectorNx; ++xIndex)
  {
    const G4double x = kDetectorPixelX * (xIndex - kDetectorNx / 2.0 - 0.5);
    for (G4int zIndex = 1; zIndex <= kDetectorNz; ++zIndex)
    {
      const G4double z = kDetectorPixelZ * (zIndex - kDetectorNz / 2.0 - 0.5);
      new G4PVPlacement(nullptr,
                        G4ThreeVector(x * mm, kDetectorCenterY * mm, z * mm),
                        fScintillatorLV, "Scin", worldLV, false, copyNumber++, false);
    }
  }
  Require(copyNumber - 1 == kDetectorCount, "EHE015", "Placed detector count is not 2312.");

  auto* collimatorVis = new G4VisAttributes();
  collimatorVis->SetForceSolid(true);
  collimatorLV->SetVisAttributes(collimatorVis);
  auto* detectorVis = new G4VisAttributes();
  detectorVis->SetForceSolid(true);
  fScintillatorLV->SetVisAttributes(detectorVis);
  worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());

  G4cout << "EHE geometry constructed: " << kHoleCount << " holes, "
         << kDetectorCount << " NaI detector bins." << G4endl;
  G4cout << "Pb plate Y = "
         << kCollimatorCenterY - kCollimatorThicknessY / 2.0 << " .. "
         << kCollimatorCenterY + kCollimatorThicknessY / 2.0 << " mm; NaI Y = "
         << kDetectorCenterY - kDetectorThicknessY / 2.0 << " .. "
         << kDetectorCenterY + kDetectorThicknessY / 2.0 << " mm." << G4endl;
  return worldPV;
}
