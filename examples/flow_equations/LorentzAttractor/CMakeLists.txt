cmake_minimum_required(VERSION 3.18)

project(lorentzattractor VERSION 0.0.1 LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Install directories
include(GNUInstallDirs)

# Options
option(BUILD_PYTHON_BINDINGS "Enable building of Python bindings" ON)
option(USE_SYSTEM_PYBIND11 "Use system-installed pybind11" ON)

# Path to superbuild libraries - MODIFY THIS FOR YOUR SYSTEM
set(SUPERBUILD_INSTALL_DIR "" CACHE PATH "Path to superbuild installed libraries")
if(NOT SUPERBUILD_INSTALL_DIR)
    message(FATAL_ERROR "Please set SUPERBUILD_INSTALL_DIR to your ODEVisualization superbuild install directory")
endif()

# Add superbuild path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${SUPERBUILD_INSTALL_DIR}")
message(STATUS "Looking for dependencies in: ${CMAKE_PREFIX_PATH}")

# Find JSON include directory
find_path(JSON_INCLUDE_DIR "nlohmann/json.hpp"
    PATHS
        ${CMAKE_PREFIX_PATH}
        ${SUPERBUILD_INSTALL_DIR}
    PATH_SUFFIXES
        include
    NO_DEFAULT_PATH
)

# If not found in superbuild, try system paths
if(NOT JSON_INCLUDE_DIR)
    find_path(JSON_INCLUDE_DIR "nlohmann/json.hpp")
endif()

if(JSON_INCLUDE_DIR)
    message(STATUS "Found JSON headers at: ${JSON_INCLUDE_DIR}")
else()
    message(WARNING "nlohmann/json.hpp not found, compilation might fail")
endif()

# Set global CUDA flags for JSON include - this is crucial for proper compilation
if(JSON_INCLUDE_DIR)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${JSON_INCLUDE_DIR}")
    message(STATUS "Added global CUDA flag: -I${JSON_INCLUDE_DIR}")
endif()

# Find required packages
find_package(devdat REQUIRED)
find_package(flowequations REQUIRED)

message(STATUS "Found devdat: ${devdat_FOUND}")
message(STATUS "Found flowequations: ${flowequations_FOUND}")

# === Python bindings setup ===
if(BUILD_PYTHON_BINDINGS)
    if(USE_SYSTEM_PYBIND11)
        find_package(pybind11 CONFIG REQUIRED)
    else()
        # Use FetchContent for pybind11
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.10.4
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    add_subdirectory(python_pybind)
endif()

# === Main library ===
add_library(lorentzattractor STATIC
    src/lorentz_attractor_flow_equation.cu
    src/lorentz_attractor_jacobian_equation.cu
)

# Set CUDA compile options
target_compile_options(lorentzattractor PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda 
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
    >
) 

# Set library dependencies
target_link_libraries(lorentzattractor 
    PUBLIC 
        flowequations::flowequations 
        devdat::devdat
)

# Set include directories
target_include_directories(lorentzattractor 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Create alias for downstream projects
add_library(${PROJECT_NAME}::lorentzattractor ALIAS lorentzattractor)

# === Installation ===

# Install targets
install(
    TARGETS lorentzattractor
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate and install config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Export targets
install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Install config files
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Print summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)