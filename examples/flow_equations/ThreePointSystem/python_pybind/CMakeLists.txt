pybind11_add_module(threepointsystem_python
three_point_system_python.cu
)

target_link_libraries(threepointsystem_python PUBLIC threepointsystem paramhelper::paramhelper Eigen3::Eigen odevisualization::odevisualization)
set_target_properties(threepointsystem_python PROPERTIES OUTPUT_NAME threepointsystemsimulation)

if(RUN_WITH_PYTHON_BACKEND)
    target_compile_definitions(threepointsystem_python PUBLIC PYTHON_BACKEND)
endif()

if(GPU)
    target_compile_definitions(threepointsystem_python PUBLIC -D GPU)
endif()

target_compile_options(threepointsystem_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(threepointsystem_python PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
