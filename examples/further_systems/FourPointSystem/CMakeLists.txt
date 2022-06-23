cmake_minimum_required(VERSION 3.15)

# Silence warnings about empty CUDA_ARCHITECTURES properties on example targets:
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

# Set a name and a version number for your project:
project(FourPointSystem VERSION 0.0.1 LANGUAGES CXX CUDA)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# General compilation options
set(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Enable building of Python bindings")


# Include libraries
find_package(ParamHelper CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(devdat CONFIG REQUIRED)
find_package(flowequations CONFIG REQUIRED)
find_package(ODEVisualization CONFIG REQUIRED)

# Include ODE systems
find_package(FourPointSystem CONFIG REQUIRED)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)

add_executable(
    FourPointSystem
    src/main.cpp
    src/fixed_point_search_t.cu
)

target_link_libraries(FourPointSystem PUBLIC paramhelper::paramhelper flowequations::flowequations devdat::devdat Eigen3::Eigen odevisualization::odevisualization fourpointsystem::fourpointsystem)

target_compile_options(FourPointSystem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

target_include_directories(FourPointSystem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
)
