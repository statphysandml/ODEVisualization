pybind11_add_module(nonlinearsystem_python
nonlinear_system_python.cu
)

target_link_libraries(nonlinearsystem_python PUBLIC nonlinearsystem flowequations::flowequations devdat::devdat)

target_compile_options(nonlinearsystem_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(nonlinearsystem_python PROPERTIES OUTPUT_NAME nonlinearsystemsimulation)