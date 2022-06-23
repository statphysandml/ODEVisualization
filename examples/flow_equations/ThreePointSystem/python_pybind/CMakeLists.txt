pybind11_add_module(threepointsystem_python
three_point_system_python.cu
)

target_link_libraries(threepointsystem_python PUBLIC threepointsystem flowequations::flowequations devdat::devdat)

target_compile_options(threepointsystem_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(threepointsystem_python PROPERTIES OUTPUT_NAME threepointsystemsimulation)