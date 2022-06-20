add_library(lorentzattractor STATIC
    lorentz_attractor_flow_equation.cu
    lorentz_attractor_jacobian_equation.cu
)

if(GPU)
    target_compile_definitions(lorentzattractor PUBLIC -D GPU)
endif()

target_compile_options(lorentzattractor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(lorentzattractor PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_link_libraries(lorentzattractor PUBLIC paramhelper::paramhelper Eigen3::Eigen odevisualization::odevisualization)

target_include_directories(lorentzattractor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)