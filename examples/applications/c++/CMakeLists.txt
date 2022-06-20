cmake_minimum_required(VERSION 3.18)

# Set a name and a version number for your project:
project(Example VERSION 0.0.1 LANGUAGES CXX CUDA)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Include libraries
find_package(ParamHelper CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(ODEVisualization CONFIG REQUIRED)

# Include ODE systems
find_package(LorentzAttractor CONFIG REQUIRED)

SET(GPU ON CACHE BOOL "Running on GPU")

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)

add_executable(
    Example
    src/main.cpp
    src/dev_dat_t.cu
    src/fixed_point_search_t.cu
    src/flow_equations_t.cu
    src/evolution_t.cu
    src/flow_observer_t.cu
    src/separatrizes_t.cu
)

if(GPU)
    target_compile_definitions(Example PUBLIC -D GPU)
endif()

target_compile_options(Example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(Example PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_link_libraries(Example PUBLIC odevisualization::odevisualization lorentzattractor::lorentzattractor)

target_include_directories(Example PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
)
