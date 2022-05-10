cmake_minimum_required(VERSION 3.15)

# Set a name and a version number for your project:
project(ThreePointSystem VERSION 0.0.1 LANGUAGES CXX CUDA)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# General compilation options
### set(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Enable building of Python bindings")
set(RUN_WITH_PYTHON_BACKEND ON CACHE BOOL "Use Python backend to compute expectation values, etc.")

# Project specific compliation options (used by config.h.in)
set(CLUSTER_MODE "local" CACHE STRING "Mode to run the cluster ('local' or 'on_cluster').")
set(PYTHON_SCRIPTS_PATH "./python_scripts" CACHE STRING "Path to the potential python scripts evaluated by the python backend.")

# Include libraries
find_package(ParamHelper CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(ODEVisualizationLib CONFIG REQUIRED)

# SET(CudaUsage "OFF" CACHE STRING "Defines whether code is executed on GPU or CPU with thrust.")
SET(GPU ON CACHE BOOL "Running on GPU")

configure_file(./include/config.h.in ../include/config.h @ONLY)

if(RUN_WITH_PYTHON_BACKEND)
    set(target_link_libraries_python_appendix "pybind11::embed")
    message("Running with python backend enabled.")
else()
    # Use library without the integration of python code
    message("Running with python backend disabled.")
endif()

if((BUILD_PYTHON_BINDINGS OR RUN_WITH_PYTHON_BACKEND) AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11")
    find_package(Git QUIET)
    if(GIT_FOUND)
        message("Downloading pybind11 into ${CMAKE_CURRENT_SOURCE_DIR}/ext")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext")
        execute_process(COMMAND ${GIT_EXECUTABLE} clone https://github.com/pybind/pybind11.git
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext"
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git clone https://github.com/pybind/pybind11.git failed with ${GIT_SUBMOD_RESULT}.")
        endif()
        execute_process(COMMAND ${GIT_EXECUTABLE} checkout "v2.6.2"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11"
                RESULT_VARIABLE GIT_SUBMOD_RESULT OUTPUT_QUIET ERROR_QUIET)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git checkout \"v2.6.2\" failed with ${GIT_SUBMOD_RESULT}.")
        endif()
    endif()

    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/ext/pybind11")
        message(FATAL_ERROR "pybind11 was not added to ${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11. Please install git or add pybind11 manually.")
    endif()
endif()

### if(BUILD_PYTHON_BINDINGS)
###     message("Integrate python for python_bindings")
###     add_subdirectory(ext/pybind11)
###     # Python wrapper
###     # add_subdirectory(python_pybind)
if(RUN_WITH_PYTHON_BACKEND)
###     message("Load pybind11 module")
    add_subdirectory(ext/pybind11)
endif()

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)

add_executable(
    ThreePointSystem
    src/main.cpp
    src/fixed_point_search_t.cu
    flow_equations/three_point_system/three_point_system_flow_equation.cu
    flow_equations/three_point_system/three_point_system_jacobian_equation.cu
)

if(RUN_WITH_PYTHON_BACKEND)
    target_compile_definitions(ThreePointSystem PUBLIC PYTHON_BACKEND)
endif()

if(GPU)
    target_compile_definitions(ThreePointSystem PUBLIC -D GPU)
endif()

target_compile_options(ThreePointSystem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 
set_target_properties(ThreePointSystem PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_link_libraries(ThreePointSystem PUBLIC odevisualization::odevisualization ${target_link_libraries_python_appendix})
