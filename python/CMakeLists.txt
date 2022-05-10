pybind11_add_module(odevisualization_python odevisualization_python.cu)

target_link_libraries(odevisualization_python PUBLIC odevisualization paramhelper::paramhelper Eigen3::Eigen)
set_target_properties(odevisualization_python PROPERTIES OUTPUT_NAME odevisualization)

if(RUN_WITH_PYTHON_BACKEND)
    target_compile_definitions(odevisualization_python PUBLIC PYTHON_BACKEND)
endif()

if(GPU)
    target_compile_definitions(odevisualization_python PUBLIC -D GPU)
endif()

target_compile_options(odevisualization_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(odevisualization_python PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
