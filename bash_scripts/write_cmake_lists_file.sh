cat >$path_to_project_root/$project_name/cmake/CMakeLists.txt <<EOL
cmake_minimum_required(VERSION 3.0.2)

project(ODESolver)

# https://gist.github.com/erikzenker/713c4ff76949058d5d5d
# https://codeyarns.com/2013/09/13/how-to-build-cuda-programs-using-cmake/

# https://github.com/thrust/thrust/wiki/Device-Backends

# Pass options to NVCC

# nvcc -O2 main.cu -o main -std=c++11 --expt-extended-lambda -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
# https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
# https://devblogs.nvidia.com/building-cuda-applications-cmake/

# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_PROPAGATE_HOST_FLAGS OFF)
# set(CUDA_HOST_COMPILER clang++)


# Links to Cuda arch gen
# https://stackoverflow.com/questions/35656294/cuda-how-to-use-arch-and-code-and-sm-vs-compute

# Cuda compatiblity
# https://devtalk.nvidia.com/default/topic/1028960/cuda-setup-and-installation/looking-for-cuda-compatibility-chart-for-nvidia-drivers/
# https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690#30820690
# https://stackoverflow.com/questions/28932864/cuda-compute-capability-requirements/28933055#28933055
# https://arnon.dk/check-cuda-installed/

# Get cuda driver version
# nvidia-smi

# Cuda
set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda-10.1")
FIND_PACKAGE(CUDA QUIET REQUIRED)
if(CUDA_FOUND)
    message("Cuda = \${CUDA_INCLUDE_DIRS}")
endif()

# Uni
set(BOOST_ROOT "/opt/boost_1_70_0")

# Boost
message("\${BOOST_ROOT}")
# FIND_PACKAGE( Boost 1.71 REQUIRED COMPONENTS filesystem)
if(Boost_FOUND)
    include_directories(\${Boost_INCLUDE_DIRS})
    message("Boost = \${Boost_INCLUDE_DIRS}")
endif()

# Python

# Home
# set(PYTHON_LIBRARIES "/home/lukas/.miniconda3/envs/flowequation/lib/libpython3.7m.so")
# set(PYTHON_EXECUTABLE "/home/lukas/.miniconda3/envs/flowequation/bin/python3.7m")
# include_directories("/home/lukas/.miniconda3/envs/flowequation/include/python3.7m")
# find_package(PythonInterp 3 REQUIRED)
# find_package(PythonLibs 3 REQUIRED)
# include_directories(\${PYTHON_INCLUDE_DIRS})

# Uni
set(PYTHON_LIBRARIES "/home/kades/.conda/envs/pytorchlocal3/lib/libpython3.7m.so")
set(PYTHON_EXECUTABLE "/home/kades/.conda/envs/pytorchlocal3/bin/python3.7m")
include_directories("/home/kades/.conda/envs/pytorchlocal3/include/python3.7m")
# find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
include_directories(\${PYTHON_INCLUDE_DIRS})

message("\${PYTHON_EXECUTABLE}")


# COMPILE CU FILES
file(GLOB CUDA_FILES "src/" *.cu)
# list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75; --expt-extended-lambda; --expt-relaxed-constexpr") #  -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp"
list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60; --expt-extended-lambda; --expt-relaxed-constexpr") #  -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp"
CUDA_COMPILE(CU_O \${CUDA_FILES})

set(CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} -std=c++14 -static-libstdc++ -lboost_system") # -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp

cuda_add_executable(ODESolver
        ../${path_to_program_directory}/ode_solver/src/flow_equation_interface/flow_equation_system.cu
        ../${path_to_program_directory}/ode_solver/src/executer.cu
        ../${path_to_program_directory}/ode_solver/src/hypercubes/hypercubes.cu
        ../${path_to_program_directory}/ode_solver/src/hypercubes/node.cu
        ../${path_to_program_directory}/ode_solver/src/hypercubes/buffer.cu
        ../${path_to_program_directory}/ode_solver/src/hypercubes/nodesexpander.cu
        ../${path_to_program_directory}/ode_solver/src/util/helper_functions.cu
        ../${path_to_program_directory}/ode_solver/src/util/monitor.cu
        ../${path_to_program_directory}/ode_solver/src/util/dev_dat.cu
        ../${path_to_program_directory}/ode_solver/src/util/lambda_range_generator.cu
        ../${path_to_program_directory}/ode_solver/src/extern/parameters.cu
        ../${path_to_program_directory}/ode_solver/src/extern/pathfinder.cu
        ../${path_to_program_directory}/ode_solver/src/coordinate_operator.cu
        ../${path_to_program_directory}/ode_solver/src/visualization.cu
        ../${path_to_program_directory}/ode_solver/src/observers/evolution.cu
        ../${path_to_program_directory}/ode_solver/src/observers/conditional_range_observer.cu
        ../${path_to_program_directory}/ode_solver/src/observers/conditional_intersection_observer.cu
        ../${path_to_program_directory}/ode_solver/src/observers/tracking_observer.cu
        ../${path_to_program_directory}/ode_solver/src/fixed_point_search.cu
        ../${path_to_program_directory}/ode_solver/../examples/tests.cu

        ../main.cu
        ../flow_equations/flow_equation.cu
        ../flow_equations/jacobian.cu
        )

target_link_libraries(ODESolver \${PYTHON_LIBRARIES})

option( GPU "Enable GPU" OFF )

if( GPU )
    target_compile_definitions(ODESolver PUBLIC -D GPU)
endif()

# TARGET_LINK_LIBRARIES( ODESolver LINK_PUBLIC \${Boost_LIBRARIES} )


# call within different folders:
# cmake ../cmake/ -DCMAKE_BUILD_TYPE=Debug
# cmake ../cmake/ -DCMAKE_BUILD_TYPE=Release
# cmake ../cmake/ -DCMAKE_BUILD_TYPE=Debug -DGPU=ON
# cmake ../cmake/ -DCMAKE_BUILD_TYPE=Release -DGPU=ON

EOL
