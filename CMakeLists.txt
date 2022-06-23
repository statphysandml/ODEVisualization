cmake_minimum_required(VERSION 3.15)

# Silence warnings about empty CUDA_ARCHITECTURES properties on example targets:
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

# Set a name and a version number for your project:
project(ODEVisualization VERSION 0.0.1 LANGUAGES CXX CUDA)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-lstdc++fs")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation options
set(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Enable building of Python bindings")
set(BUILD_DOCS ON CACHE BOOL "Enable building of documentation")

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/ext/ParamHelper/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/ext/FlowEquationInterface/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Compilation options

# compile the external submodule
include_directories(ext/ParamHelper/include)
add_subdirectory(ext/ParamHelper)

include_directories(ext/FlowEquationInterface/ext/DevDat/include)
include_directories(ext/FlowEquationInterface/include)
add_subdirectory(ext/FlowEquationInterface)

include_directories(ext/eigen/Eigen)
add_subdirectory(ext/eigen)

if(BUILD_PYTHON_BINDINGS)
    message("Integrate python for python_bindings")
    # Add Python bindings
    add_subdirectory(ext/pybind11)
    # Python wrapper
    add_subdirectory(python)
endif()

# compile the library
add_subdirectory(src)


# compile the tests
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(BUILD_DOCS)
  # Add the documentation
  add_subdirectory(doc)
endif()


# Add an alias target for use if this project is included as a subproject in another project
add_library(odevisualization::odevisualization ALIAS odevisualization)

# Install targets and configuration
install(
  TARGETS odevisualization
  EXPORT odevisualization-config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT odevisualization-config
  NAMESPACE odevisualization::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/odevisualization
)

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
