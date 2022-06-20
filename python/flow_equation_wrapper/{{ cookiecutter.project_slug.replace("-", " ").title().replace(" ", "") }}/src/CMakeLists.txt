add_library({{ cookiecutter.project_slug.replace("-", "") }} STATIC
    {{ cookiecutter.project_slug.replace("-", "_") }}_flow_equation.cu
    {{ cookiecutter.project_slug.replace("-", "_") }}_jacobian_equation.cu
)

if(GPU)
    target_compile_definitions({{ cookiecutter.project_slug.replace("-", "") }} PUBLIC -D GPU)
endif()

target_compile_options({{ cookiecutter.project_slug.replace("-", "") }} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties({{ cookiecutter.project_slug.replace("-", "") }} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_link_libraries({{ cookiecutter.project_slug.replace("-", "") }} PUBLIC paramhelper::paramhelper Eigen3::Eigen odevisualization::odevisualization)

target_include_directories({{ cookiecutter.project_slug.replace("-", "") }} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)