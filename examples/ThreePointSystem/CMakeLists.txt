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
set(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Enable building of Python bindings")


# Include libraries
find_package(ParamHelper CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(ODEVisualization CONFIG REQUIRED)

SET(GPU ON CACHE BOOL "Running on GPU")

if(BUILD_PYTHON_BINDINGS AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11")
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

if(BUILD_PYTHON_BINDINGS AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11_json")
    find_package(Git QUIET)
    if(GIT_FOUND)
        message("Downloading pybind11_json into ${CMAKE_CURRENT_SOURCE_DIR}/ext")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext")
        execute_process(COMMAND ${GIT_EXECUTABLE} clone https://github.com/pybind/pybind11_json.git
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext"
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "https://github.com/pybind/pybind11_json.git failed with ${GIT_SUBMOD_RESULT}.")
        endif()
        execute_process(COMMAND ${GIT_EXECUTABLE} checkout "0.2.12"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11_json"
                RESULT_VARIABLE GIT_SUBMOD_RESULT OUTPUT_QUIET ERROR_QUIET)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git checkout \"0.2.12\" failed with ${GIT_SUBMOD_RESULT}.")
        endif()
    endif()

    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/ext/pybind11_json")
        message(FATAL_ERROR "pybind11_json was not added to ${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11_json. Please install git or add pybind11_json manually.")
    endif()
endif()

if(BUILD_PYTHON_BINDINGS)
    message("Integrate python for python_bindings")
    add_subdirectory(ext/pybind11)
    # add_subdirectory(ext/pybind11_json)

    # Python wrapper
    add_subdirectory(python_pybind)
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

if(GPU)
    target_compile_definitions(ThreePointSystem PUBLIC -D GPU)
endif()

target_compile_options(ThreePointSystem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(ThreePointSystem PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_link_libraries(ThreePointSystem PUBLIC odevisualization::odevisualization)

message("Source dir ${CMAKE_SOURCE_DIR}")

target_include_directories(ThreePointSystem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
)
