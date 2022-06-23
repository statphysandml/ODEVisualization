add_library(threepointsystem STATIC
    three_point_system_flow_equation.cu
    three_point_system_jacobian_equation.cu
)

target_compile_options(threepointsystem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

target_link_libraries(threepointsystem PUBLIC flowequations::flowequations devdat::devdat)

target_include_directories(threepointsystem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)