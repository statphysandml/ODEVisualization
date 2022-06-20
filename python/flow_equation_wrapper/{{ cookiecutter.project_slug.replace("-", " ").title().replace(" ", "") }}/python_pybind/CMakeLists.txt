pybind11_add_module({{ cookiecutter.project_slug.replace("-", "") }}_python
{{ cookiecutter.project_slug.replace("-", "_") }}_python.cu
)

target_link_libraries({{ cookiecutter.project_slug.replace("-", "") }}_python PUBLIC {{ cookiecutter.project_slug.replace("-", "") }} paramhelper::paramhelper Eigen3::Eigen odevisualization::odevisualization)
set_target_properties({{ cookiecutter.project_slug.replace("-", "") }}_python PROPERTIES OUTPUT_NAME {{ cookiecutter.project_slug.replace("-", "") }}simulation)

if(RUN_WITH_PYTHON_BACKEND)
    target_compile_definitions({{ cookiecutter.project_slug.replace("-", "") }}_python PUBLIC PYTHON_BACKEND)
endif()

if(GPU)
    target_compile_definitions({{ cookiecutter.project_slug.replace("-", "") }}_python PUBLIC -D GPU)
endif()

target_compile_options({{ cookiecutter.project_slug.replace("-", "") }}_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties({{ cookiecutter.project_slug.replace("-", "") }}_python PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
