cmake_minimum_required(VERSION 3.18)
project(odevisualization VERSION 1.0.0 LANGUAGES CXX CUDA)

# Standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Disable LTO to avoid compiler version conflicts
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(USE_SYSTEM_PYBIND11 "Use system-installed pybind11" OFF)

# Enhanced debugging for dependency paths
message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

# Find dependencies
find_package(Boost REQUIRED)
find_package(devdat REQUIRED)
find_package(paramhelper REQUIRED)
find_package(flowequations REQUIRED)

# Look for JSON headers explicitly - helps with CUDA compilation
find_path(JSON_INCLUDE_DIR "nlohmann/json.hpp"
    PATHS
        ${CMAKE_PREFIX_PATH}
        ${SUPERBUILD_INSTALL_DIR}
        ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES
        include
)

if(JSON_INCLUDE_DIR)
    message(STATUS "Found JSON headers at: ${JSON_INCLUDE_DIR}")
else()
    message(WARNING "nlohmann/json.hpp not found directly, compilation might fail")
endif()

add_subdirectory(src)

# Create an alias target
add_library(${PROJECT_NAME}::odevisualization ALIAS odevisualization)

# Installation
include(GNUInstallDirs)

# Install headers
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install targets
install(
    TARGETS odevisualization
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create package config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/odevisualizationConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install config files
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Add subdirectories based on options
if(BUILD_PYTHON_BINDINGS)
    # Handle pybind11 dependency
    if(USE_SYSTEM_PYBIND11)
        find_package(pybind11 CONFIG REQUIRED)
    else()
        # Use provided pybind11 or fetch it
        if(DEFINED PYBIND11_SOURCE_DIR AND EXISTS "${PYBIND11_SOURCE_DIR}")
            add_subdirectory(${PYBIND11_SOURCE_DIR} pybind11)
        else()
            include(FetchContent)
            FetchContent_Declare(
                pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG v2.10.4
            )
            FetchContent_MakeAvailable(pybind11)
        endif()
    endif()
    
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()