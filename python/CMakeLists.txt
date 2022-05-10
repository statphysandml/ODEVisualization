pybind11_add_module(odevisualisation_python odevisualisation_python.cu)

target_link_libraries(odevisualisation_python PUBLIC odevisualisation paramhelper::paramhelper Eigen3::Eigen)
set_target_properties(odevisualisation_python PROPERTIES OUTPUT_NAME odevisualisation)

if(RUN_WITH_PYTHON_BACKEND)
    target_compile_definitions(odevisualisation_python PUBLIC PYTHON_BACKEND)
endif()

if(GPU)
    target_compile_definitions(odevisualisation_python PUBLIC -D GPU)
endif()

target_compile_options(odevisualisation_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(odevisualisation_python PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
