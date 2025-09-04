cmake_minimum_required(VERSION 3.15)

# Silence warnings about empty CUDA_ARCHITECTURES properties on example targets:
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

# Set a name and a version number for your project:
project(Example VERSION 0.0.1 LANGUAGES CXX CUDA)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable CUDA device linking - crucial for libraries with device code
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
message(STATUS "Enabled CUDA separable compilation for device linking")

# Path to superbuild libraries
set(SUPERBUILD_INSTALL_DIR "" CACHE PATH "Path to superbuild installed libraries")
if(NOT SUPERBUILD_INSTALL_DIR)
    message(STATUS "SUPERBUILD_INSTALL_DIR not set, will use system paths")
endif()

# Add superbuild path to CMAKE_PREFIX_PATH if provided
if(SUPERBUILD_INSTALL_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${SUPERBUILD_INSTALL_DIR}")
    message(STATUS "Looking for dependencies in: ${CMAKE_PREFIX_PATH}")
endif()

# Debug: Print the actual paths being searched
message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

# Find JSON include directory for CUDA compilation
find_path(JSON_INCLUDE_DIR "nlohmann/json.hpp"
    PATHS
        ${CMAKE_PREFIX_PATH}
        ${SUPERBUILD_INSTALL_DIR}
    PATH_SUFFIXES
        include
    NO_DEFAULT_PATH
)

# If not found in specified paths, try system paths
if(NOT JSON_INCLUDE_DIR)
    find_path(JSON_INCLUDE_DIR "nlohmann/json.hpp")
endif()

if(JSON_INCLUDE_DIR)
    message(STATUS "Found JSON headers at: ${JSON_INCLUDE_DIR}")
    
    # Add JSON include path to global CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${JSON_INCLUDE_DIR}")
    message(STATUS "Added global CUDA flag: -I${JSON_INCLUDE_DIR}")
else()
    message(WARNING "nlohmann/json.hpp not found, compilation might fail")
endif()

# Find required packages
find_package(Boost REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(paramhelper CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(devdat CONFIG REQUIRED)
find_package(flowequations CONFIG REQUIRED)
find_package(odevisualization CONFIG REQUIRED)
find_package(lorentzattractor CONFIG REQUIRED)

# Print a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)

# Define source files for the Example executable
set(EXAMPLE_SOURCES
    src/main.cpp
    src/dev_dat_t.cu
    src/fixed_point_search_t.cu
    src/flow_equations_t.cu
    src/evolution_t.cu
    src/flow_observer_t.cu
    src/separatrizes_t.cu
)

# Create an object library first to handle CUDA device linking correctly
add_library(Example_obj OBJECT ${EXAMPLE_SOURCES})

# Set CUDA compile options for the object library
target_compile_options(Example_obj PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
    >
) 

# Set include directories for the object library
target_include_directories(Example_obj PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/include/
    $<$<BOOL:${JSON_INCLUDE_DIR}>:${JSON_INCLUDE_DIR}>
)

# Link libraries to the object library
target_link_libraries(Example_obj PUBLIC 
    paramhelper::paramhelper
    flowequations::flowequations 
    devdat::devdat 
    odevisualization::odevisualization 
    lorentzattractor::lorentzattractor
    Eigen3::Eigen
)

# Create the actual executable with device linking
add_executable(Example $<TARGET_OBJECTS:Example_obj>)

# Link the executable with the CUDA device code
target_link_libraries(Example PUBLIC Example_obj)

# Ensure CUDA device linking happens
set_target_properties(Example PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Install the executable
install(
    TARGETS Example
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)