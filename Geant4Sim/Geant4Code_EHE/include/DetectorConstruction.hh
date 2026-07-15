#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

#include <vector>

class G4LogicalVolume;
class G4Material;

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction() override = default;

    G4VPhysicalVolume* Construct() override;

    G4int Getnx() const { return kDetectorNz; }
    G4int Getny() const { return kDetectorNx; }
    G4int Getlayers() const { return 1; }
    G4int GetScinNum() const { return kDetectorCount; }

    // Kept for compatibility with the shared Run/Event/Stepping actions.
    G4int GetScinInfo(G4int, G4int) const { return 1; }
    G4double GetSize() const { return kDetectorPixelX; }
    G4double GetLength() const { return kDetectorThicknessY; }

    static constexpr G4int kDetectorNx = 68;
    static constexpr G4int kDetectorNz = 34;
    static constexpr G4int kDetectorCount = kDetectorNx * kDetectorNz;
    static constexpr G4int kHoleRows = 25;
    static constexpr G4int kHoleColumns = 50;
    static constexpr G4int kHoleCount = kHoleRows * kHoleColumns;

  private:
    struct HoleCenter
    {
      G4double x;
      G4double z;
    };

    void DefineMaterials();
    std::vector<HoleCenter> BuildHoleCenters() const;
    void ValidateGeometry(const std::vector<HoleCenter>& holes) const;
    void WriteGeometryFiles(const std::vector<HoleCenter>& holes) const;

    G4Material* fVacuum;
    G4Material* fLead;
    G4Material* fNaI;
    G4LogicalVolume* fScintillatorLV;

    // Values match FileGenerater_3D_Unified/config_geometry.m and
    // build_collimator.m for ConventionalSPECT.
    static constexpr G4double kFovCenterY = -245.0;
    static constexpr G4double kCollimatorSizeX = 330.0;
    static constexpr G4double kCollimatorThicknessY = 50.5;
    static constexpr G4double kCollimatorSizeZ = 165.0;
    // Shared with the front face of the first JSCC detector layer.
    static constexpr G4double kCommonFrontFaceDistance = 198.5;
    static constexpr G4double kFovToCollimatorOrigin =
      kCommonFrontFaceDistance + kCollimatorThicknessY / 2.0;
    static constexpr G4double kCollimatorCenterY = kFovCenterY + kFovToCollimatorOrigin;
    static constexpr G4double kHoleDiameter = 2.5;
    static constexpr G4double kSeptalThickness = 3.4;
    static constexpr G4double kHolePitch = kHoleDiameter + kSeptalThickness;
    static constexpr G4double kDetectorPixelX = 4.0;
    static constexpr G4double kDetectorThicknessY = 10.0;
    static constexpr G4double kDetectorPixelZ = 4.0;
    static constexpr G4double kDetectorCenterY =
      kCollimatorCenterY + kCollimatorThicknessY / 2.0 + kDetectorThicknessY / 2.0;
};

#endif
