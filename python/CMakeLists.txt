# Silence warnings about empty CUDA_ARCHITECTURES properties on example targets:
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

pybind11_add_module(odevisualization_python
    src/dev_dat_python.cu
    src/evolution_python.cu
    src/flow_equations_python.cu
    src/modes_python.cu
    src/odevisualization_python.cu
    src/recursive_search_python.cu
)

target_link_libraries(odevisualization_python PUBLIC odevisualization paramhelper::paramhelper Eigen3::Eigen)

target_compile_options(odevisualization_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(odevisualization_python PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(odevisualization_python PROPERTIES OUTPUT_NAME odevisualizationlib)
