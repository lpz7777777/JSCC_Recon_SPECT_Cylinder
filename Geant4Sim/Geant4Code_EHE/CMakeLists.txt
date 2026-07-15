cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(ehe_spect LANGUAGES CXX)

option(WITH_GEANT4_UIVIS "Build with Geant4 UI and visualization drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

include(${Geant4_USE_FILE})

set(EHE_SOURCES
  ehe_spect.cc
  src/ActionInitialization.cc
  src/DetectorConstruction.cc
  src/EventAction.cc
  src/PrimaryGeneratorAction.cc
  src/Run.cc
  src/RunAction.cc
  src/SteppingAction.cc
  src/SteppingVerbose.cc
)

add_executable(ehe_spect ${EHE_SOURCES})
target_include_directories(ehe_spect PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(ehe_spect PRIVATE cxx_std_17)
target_link_libraries(ehe_spect PRIVATE ${Geant4_LIBRARIES})

# Geant4 11 on Windows exports driver-specific definitions such as
# G4UI_USE_WIN32 and G4VIS_USE_OPENGLWIN32.  Use one project-level switch in
# the entry point so the interactive branch is compiled whenever UI/Vis was
# requested, independent of the platform-specific Geant4 macro names.
if(WITH_GEANT4_UIVIS)
  target_compile_definitions(ehe_spect PRIVATE EHE_USE_UIVIS)
endif()

# Keep the executable beside the copied macros.  This makes build/ehe_spect.exe
# the canonical Windows executable and keeps direct Explorer launches useful.
set_target_properties(ehe_spect PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_BINARY_DIR}"
)

set(EHE_MACROS
  example_225ac_point.mac
  smoke_218.mac
  smoke_440.mac
  vis.mac
)
foreach(macro ${EHE_MACROS})
  configure_file(${PROJECT_SOURCE_DIR}/${macro} ${PROJECT_BINARY_DIR}/${macro} COPYONLY)
endforeach()

install(TARGETS ehe_spect DESTINATION bin)
