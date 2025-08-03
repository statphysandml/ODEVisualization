add_library({{ cookiecutter.project_slug.replace("-", "") }} STATIC
    {{ cookiecutter.project_slug.replace("-", "_") }}_flow_equation.cu
    {{ cookiecutter.project_slug.replace("-", "_") }}_jacobian_equation.cu
)

target_compile_options({{ cookiecutter.project_slug.replace("-", "") }} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

target_link_libraries({{ cookiecutter.project_slug.replace("-", "") }} PUBLIC flowequations::flowequations devdat::devdat)

target_include_directories({{ cookiecutter.project_slug.replace("-", "") }} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)