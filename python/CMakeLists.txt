pybind11_add_module(odevisualization_python
    src/dev_dat_python.cu
    src/evolution_python.cu
    src/flow_equations_python.cu
    src/modes_python.cu
    src/odevisualization_python.cu
    src/recursive_search_python.cu
)

target_link_libraries(odevisualization_python PUBLIC odevisualization stdc++fs paramhelper::paramhelper flowequations::flowequations devdat::devdat Eigen3::Eigen)

target_compile_options(odevisualization_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(odevisualization_python PROPERTIES OUTPUT_NAME odevisualizationlib)
