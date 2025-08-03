cmake_minimum_required(VERSION 3.18)

project(ODEVisualization VERSION 0.0.1 LANGUAGES CXX CUDA)

# Include necessary modules
include(GNUInstallDirs)
include(ExternalProject)
include(FetchContent)

# Build options
option(BUILD_PYTHON_BINDINGS "Enable building of Python bindings" ON)
option(BUILD_DOCS "Enable building of documentation" ON)
option(BUILD_TESTING "Build tests" ON)

# Component library options
option(USE_SYSTEM_DEVDAT "Use system-installed devdat" OFF)
option(USE_SYSTEM_FLOWEQUATIONS "Use system-installed flowequations" OFF)
option(USE_SYSTEM_PARAMHELPER "Use system-installed paramhelper" OFF)
option(USE_SYSTEM_EIGEN "Use system-installed Eigen" OFF)
option(USE_SYSTEM_PYBIND11 "Use system-installed pybind11" OFF)

# Set standard C++ properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Setup install directories for the external projects
set(SUPERBUILD_INSTALL_DIR ${CMAKE_BINARY_DIR}/installed)
set(CMAKE_PREFIX_PATH ${SUPERBUILD_INSTALL_DIR} ${CMAKE_PREFIX_PATH})

# Find Boost (required)
find_package(Boost REQUIRED)
find_package(CUDAToolkit REQUIRED)

# === Dependency Handling ===

# nlohmann_json (needed by paramhelper)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

# Store the nlohmann/json path to pass to dependencies
get_target_property(JSON_INCLUDE_DIR nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)

# 1. devdat library
if(USE_SYSTEM_DEVDAT)
    find_package(devdat REQUIRED)
else()
    ExternalProject_Add(
        devdat_ext
        GIT_REPOSITORY https://github.com/statphysandml/devdat.git
        GIT_TAG main
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${SUPERBUILD_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_TESTING=OFF
            -DDEVDAT_USE_SYSTEM_JSON=ON
            -DJSON_INCLUDE_DIR=${JSON_INCLUDE_DIR}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
    )
    set(DEVDAT_DIR ${SUPERBUILD_INSTALL_DIR})
endif()

# 2. flowequations library (depends on devdat)
if(USE_SYSTEM_FLOWEQUATIONS)
    find_package(flowequations REQUIRED)
else()
    ExternalProject_Add(
        flowequations_ext
        GIT_REPOSITORY https://github.com/statphysandml/FlowEquationInterface.git
        GIT_TAG main
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${SUPERBUILD_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_TESTING=OFF
            -DBUILD_EXAMPLES=OFF
            -DUSE_SYSTEM_JSON=ON
            -DJSON_INCLUDE_DIR=${JSON_INCLUDE_DIR}
            -DCMAKE_PREFIX_PATH=${SUPERBUILD_INSTALL_DIR}
            -DDEVDAT_DIR=${SUPERBUILD_INSTALL_DIR}
        DEPENDS devdat_ext
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
    )
    set(FLOWEQUATIONS_DIR ${SUPERBUILD_INSTALL_DIR})
endif()

# 3. paramhelper library
if(USE_SYSTEM_PARAMHELPER)
    find_package(paramhelper REQUIRED)
else()
    ExternalProject_Add(
        paramhelper_ext
        GIT_REPOSITORY https://github.com/statphysandml/paramhelper.git
        GIT_TAG main
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${SUPERBUILD_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_TESTING=OFF
            -DCMAKE_PREFIX_PATH=${SUPERBUILD_INSTALL_DIR}
            -DPARAMHELPER_USE_SYSTEM_JSON=ON
            -DJSON_INCLUDE_DIR=${JSON_INCLUDE_DIR}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
    )
    set(PARAMHELPER_DIR ${SUPERBUILD_INSTALL_DIR})
endif()

# 4. Eigen
if(USE_SYSTEM_EIGEN)
    find_package(Eigen3 REQUIRED)
    # Use the existing target from find_package
    set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})
else()
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    # Just populate, don't make available (to avoid target conflicts)
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()
    
    # Create our wrapper target with a unique name
    add_library(eigen_wrapper INTERFACE)
    target_include_directories(eigen_wrapper INTERFACE 
        $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/eigen>
    )
    
    # Store the include directory to pass to the main project
    set(EIGEN_INCLUDE_DIR ${eigen_SOURCE_DIR})
endif()

# 5. pybind11 (if needed)
if(BUILD_PYTHON_BINDINGS)
    if(USE_SYSTEM_PYBIND11)
        find_package(pybind11 REQUIRED)
    else()
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.10.4
        )
        FetchContent_MakeAvailable(pybind11)
        
        # Store the directory for the main project to use
        set(PYBIND11_SOURCE_DIR ${pybind11_SOURCE_DIR})
    endif()
endif()

# === Main ODEVisualization Project ===

# Create empty dependency list
set(ODEVIS_DEPS "")

# Add dependencies conditionally
if(NOT USE_SYSTEM_DEVDAT)
    list(APPEND ODEVIS_DEPS "devdat_ext")
endif()

if(NOT USE_SYSTEM_FLOWEQUATIONS)
    list(APPEND ODEVIS_DEPS "flowequations_ext")
endif()

if(NOT USE_SYSTEM_PARAMHELPER)
    list(APPEND ODEVIS_DEPS "paramhelper_ext")
endif()

# Use the dependency list in ExternalProject_Add
ExternalProject_Add(
    odevisualization_main
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/odevis
    BINARY_DIR ${CMAKE_BINARY_DIR}/odevis
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH=${SUPERBUILD_INSTALL_DIR}
        -DBUILD_PYTHON_BINDINGS=${BUILD_PYTHON_BINDINGS}
        -DBUILD_DOCS=${BUILD_DOCS}
        -DBUILD_TESTING=${BUILD_TESTING}
        -DUSE_SYSTEM_EIGEN=${USE_SYSTEM_EIGEN}
        -DUSE_SYSTEM_PYBIND11=${USE_SYSTEM_PYBIND11}
        -DEigen3_DIR=${Eigen3_DIR}
        -Dpybind11_DIR=${pybind11_DIR}
        -DPYBIND11_SOURCE_DIR=${PYBIND11_SOURCE_DIR}
        -DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}
        -DJSON_INCLUDE_DIR=${JSON_INCLUDE_DIR}  # Add JSON include path
    DEPENDS ${ODEVIS_DEPS}
)

# Make installation of the main project be the installation of the superbuild
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/odevis)")

# Feature summary
include(FeatureSummary)
feature_summary(WHAT ALL)