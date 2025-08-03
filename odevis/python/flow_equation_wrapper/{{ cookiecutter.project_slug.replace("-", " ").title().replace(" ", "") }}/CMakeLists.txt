cmake_minimum_required(VERSION 3.15)

# Silence warnings about empty CUDA_ARCHITECTURES properties on example targets:
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

# Set a name and a version number for your project:
project({{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }} VERSION 0.0.1 LANGUAGES CXX CUDA)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation options
set(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Enable building of Python bindings")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/FlowEquationInterface")
find_package(Git QUIET)
    if(GIT_FOUND)
        message("Downloading FlowEquationInterface into ${CMAKE_CURRENT_SOURCE_DIR}/ext")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext")
        execute_process(COMMAND ${GIT_EXECUTABLE} clone https://github.com/statphysandml/FlowEquationInterface.git
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext"
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git clone https://github.com/statphysandml/FlowEquationInterface.git failed with ${GIT_SUBMOD_RESULT}.")
        endif()
    endif()
endif()

include_directories(ext/FlowEquationInterface/ext/DevDat/include)
include_directories(ext/FlowEquationInterface/include)
add_subdirectory(ext/FlowEquationInterface)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/FlowEquationInterface")
    message(FATAL_ERROR "FlowEquationInterface was not added to ${CMAKE_CURRENT_SOURCE_DIR}/ext/FlowEquationInterface. Please install git or add FlowEquationInterface manually.")
endif()

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

# compile the library
add_subdirectory(src)

# Add an alias target for use if this project is included as a subproject in another project
add_library({{ cookiecutter.project_slug.replace("-", "") }}::{{ cookiecutter.project_slug.replace("-", "") }} ALIAS {{ cookiecutter.project_slug.replace("-", "") }})

# Install targets and configuration
install(
  TARGETS {{ cookiecutter.project_slug.replace("-", "") }}
  EXPORT {{ cookiecutter.project_slug.replace("-", "") }}-config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT {{ cookiecutter.project_slug.replace("-", "") }}-config
  NAMESPACE {{ cookiecutter.project_slug.replace("-", "") }}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/{{ cookiecutter.project_slug.replace("-", "") }}
)

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
