cmake_minimum_required(VERSION 3.22)
project(ViscoOgden VERSION 0.1 LANGUAGES CXX)

find_package(Sofa.Core REQUIRED)
find_package(Sofa.Simulation.Core REQUIRED)
find_package(Eigen3 REQUIRED)

set(SRC src/ViscoOgden)
set(HEADER_FILES ${SRC}/config.h.in ${SRC}/ViscoOgdenFEMForceField.h)
set(SOURCE_FILES ${SRC}/init.cpp ${SRC}/ViscoOgdenFEMForceField.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Core Sofa.Simulation.Core Eigen3::Eigen)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )
