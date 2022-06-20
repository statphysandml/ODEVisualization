add_library(threepointsystem STATIC
    three_point_system_flow_equation.cu
    three_point_system_jacobian_equation.cu
)

if(GPU)
    target_compile_definitions(threepointsystem PUBLIC -D GPU)
endif()

target_compile_options(threepointsystem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(threepointsystem PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_link_libraries(threepointsystem PUBLIC paramhelper::paramhelper Eigen3::Eigen odevisualization::odevisualization)

target_include_directories(threepointsystem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)