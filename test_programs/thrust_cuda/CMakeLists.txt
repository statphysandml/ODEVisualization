cmake_minimum_required(VERSION 3.0.2)
find_package(CUDA QUIET REQUIRED)

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

# message("${BOOST_ROOT}")
# FIND_PACKAGE( Boost 1.55 REQUIRED COMPONENTS filesystem)
# if(Boost_FOUND)
#     include_directories(${Boost_INCLUDE_DIRS})
#     message("Boost = ${Boost_INCLUDE_DIRS}")
# endif()

# COMPILE CU FILES
file(GLOB CUDA_FILES "src/" *.cu)
list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75; --expt-extended-lambda; --expt-relaxed-constexpr") #  -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp"
CUDA_COMPILE(CU_O ${CUDA_FILES})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -static-libstdc++ -lboost_system") # -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -static-libstdc++")

set(
        CUDA_64_BIT_DEVICE_CODE
        CUDA_NVCC_FLAGS_RELEASE
        ${CUDA_NVCC_FLAGS_RELEASE}
        -std=c++11 --expt-extended-lambda -O3
        -gencode arch=compute_60,code=sm_60
        # -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
)

set(
        CUDA_64_BIT_DEVICE_CODE
        CUDA_NVCC_FLAGS_DEBUG
        ${CUDA_NVCC_FLAGS_DEBUG};
        -std=c++11 --expt-extended-lambda -Og -g -Wall -Werror
        -gencode arch=compute_60,code=sm_60
        # -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
)

#message("${BOOST_ROOT}")
#FIND_PACKAGE( Boost 1.58 REQUIRED COMPONENTS filesystem)
#if(Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIRS})
#    message("Boost = ${Boost_INCLUDE_DIRS}")
#endif()

#if(CMAKE_COMPILER_IS_GNUCXX)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -g -Wall -Werror -std=c++11")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -std=c++11")
#set(CMAKE_CXX_FLAGS "-fopenmp - std=c++11")        ## Optimize
#    #    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
#endif()


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../test)


# nvcc main.cu -o main -std=c++11 -I /opt/

cuda_add_executable(ThrustCudaTest ./thrust_cuda_test.cu ./helper.cu)

option( GPU "Enable GPU" OFF )

if( GPU )
    target_compile_definitions(ThrustCudaTest PUBLIC -D GPU)
endif()
