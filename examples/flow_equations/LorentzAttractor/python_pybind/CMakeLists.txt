pybind11_add_module(lorentzattractor_python
lorentz_attractor_python.cu
)

target_compile_options(lorentzattractor_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

target_link_libraries(lorentzattractor_python
    PUBLIC
        lorentzattractor
        flowequations::flowequations
        devdat::devdat
)

set_target_properties(lorentzattractor_python PROPERTIES OUTPUT_NAME lorentzattractorsimulation)