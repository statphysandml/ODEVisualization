add_library(lorentzattractor STATIC
    lorentz_attractor_flow_equation.cu
    lorentz_attractor_jacobian_equation.cu
)

target_compile_options(lorentzattractor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

target_link_libraries(lorentzattractor PUBLIC flowequations::flowequations devdat::devdat)

target_include_directories(lorentzattractor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)