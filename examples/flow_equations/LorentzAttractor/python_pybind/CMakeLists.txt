pybind11_add_module(lorentzattractor_python
lorentz_attractor_python.cu
)

target_link_libraries(lorentzattractor_python PUBLIC lorentzattractor paramhelper::paramhelper Eigen3::Eigen odevisualization::odevisualization)
set_target_properties(lorentzattractor_python PROPERTIES OUTPUT_NAME lorentzattractorsimulation)

if(RUN_WITH_PYTHON_BACKEND)
    target_compile_definitions(lorentzattractor_python PUBLIC PYTHON_BACKEND)
endif()

if(GPU)
    target_compile_definitions(lorentzattractor_python PUBLIC -D GPU)
endif()

target_compile_options(lorentzattractor_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(lorentzattractor_python PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
