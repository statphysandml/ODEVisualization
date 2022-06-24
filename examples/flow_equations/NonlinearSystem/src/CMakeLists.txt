add_library(nonlinearsystem STATIC
    nonlinear_system_flow_equation.cu
    nonlinear_system_jacobian_equation.cu
)

target_compile_options(nonlinearsystem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

target_link_libraries(nonlinearsystem PUBLIC flowequations::flowequations devdat::devdat)

target_include_directories(nonlinearsystem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)