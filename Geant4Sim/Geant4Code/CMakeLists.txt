#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(gamma01)

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
#
include(${Geant4_USE_FILE})

#----------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${Geant4_INCLUDE_DIR})
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(gamma01 gamma01.cc ${sources} ${headers})
target_link_libraries(gamma01 ${Geant4_LIBRARIES} )

# Keep the executable beside CrystalMatrix.txt and the visualisation macros.
# This also makes a direct launch of build/gamma01.exe use the correct runtime
# working directory with Visual Studio's multi-configuration generators.
set_target_properties(gamma01 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_BINARY_DIR}"
)

# Geant4 11 on Windows exposes driver-specific definitions such as
# G4UI_USE_WIN32 and G4VIS_USE_OPENGLWIN32, not the generic G4UI_USE /
# G4VIS_USE macros used by older examples.  Define one project-level macro so
# the entry point can reliably enable its interactive UI and visualisation.
if(WITH_GEANT4_UIVIS)
  target_compile_definitions(gamma01 PRIVATE GAMMA01_USE_UIVIS)
endif()

#----------------------------------------------------------------------------
# Copy all scripts to the build directory, i.e. the directory in which we
# build gamma01 This is so that we can run the executable directly because it
# relies on these scripts being in the current working directory.
#
set(gamma01_SCRIPTS
    CrystalMatrix.txt
    debug.mac
    point.mac
    vis.mac
  )

foreach(_script ${gamma01_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
    )
endforeach()

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS gamma01 DESTINATION bin)

