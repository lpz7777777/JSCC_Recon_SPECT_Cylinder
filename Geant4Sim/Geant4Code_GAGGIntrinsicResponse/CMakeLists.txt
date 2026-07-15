cmake_minimum_required(VERSION 3.16)
project(GAGGIntrinsicResponse LANGUAGES CXX)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(gagg_intrinsic gagg_intrinsic.cc)
target_compile_features(gagg_intrinsic PRIVATE cxx_std_17)
target_link_libraries(gagg_intrinsic PRIVATE ${Geant4_LIBRARIES})

file(GLOB STUDY_MACROS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/macros/*.mac")
foreach(macro ${STUDY_MACROS})
  configure_file(${macro} ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
endforeach()
