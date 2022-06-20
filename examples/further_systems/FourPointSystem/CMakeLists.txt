cmake_minimum_required(VERSION 3.15)

# Set a name and a version number for your project:
project(FourPointSystem VERSION 0.0.1 LANGUAGES CXX CUDA)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# General compilation options
set(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Enable building of Python bindings")


# Include libraries
find_package(ParamHelper CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(ODEVisualization CONFIG REQUIRED)

# Include ODE systems
find_package(FourPointSystem CONFIG REQUIRED)

SET(GPU ON CACHE BOOL "Running on GPU")

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)

add_executable(
    FourPointSystem
    src/main.cpp
    src/fixed_point_search_t.cu
    flow_equations/four_point_system/four_point_system_flow_equation.cu
    flow_equations/four_point_system/four_point_system_jacobian_equation.cu
)

if(GPU)
    target_compile_definitions(FourPointSystem PUBLIC -D GPU)
endif()

target_compile_options(FourPointSystem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(FourPointSystem PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_link_libraries(FourPointSystem PUBLIC odevisualization::odevisualization)

target_include_directories(FourPointSystem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
)
