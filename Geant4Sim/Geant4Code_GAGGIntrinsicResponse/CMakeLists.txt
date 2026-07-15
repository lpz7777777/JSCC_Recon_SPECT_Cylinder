cmake_minimum_required(VERSION 3.16)
project(GAGGIntrinsicResponse LANGUAGES CXX)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(gagg_intrinsic gagg_intrinsic.cc)
target_compile_features(gagg_intrinsic PRIVATE cxx_std_17)
target_link_libraries(gagg_intrinsic PRIVATE ${Geant4_LIBRARIES})

# Visual Studio is a multi-config generator and otherwise places the executable
# in build/Debug or build/Release while the copied macros remain in build.
# Keep every configuration beside the macros to make batch execution reliable.
set_target_properties(gagg_intrinsic PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(gagg_intrinsic PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_CURRENT_BINARY_DIR}"
  )
endforeach()

file(GLOB STUDY_MACROS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/macros/*.mac")
foreach(macro ${STUDY_MACROS})
  configure_file(${macro} ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
endforeach()

if(WIN32)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/run_windows.ps1"
    "${CMAKE_CURRENT_BINARY_DIR}/run_windows.ps1"
    COPYONLY
  )
endif()
