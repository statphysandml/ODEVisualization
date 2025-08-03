pybind11_add_module({{ cookiecutter.project_slug.replace("-", "") }}_python
{{ cookiecutter.project_slug.replace("-", "_") }}_python.cu
)

target_link_libraries({{ cookiecutter.project_slug.replace("-", "") }}_python PUBLIC {{ cookiecutter.project_slug.replace("-", "") }} flowequations::flowequations devdat::devdat)

target_compile_options({{ cookiecutter.project_slug.replace("-", "") }}_python PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties({{ cookiecutter.project_slug.replace("-", "") }}_python PROPERTIES OUTPUT_NAME {{ cookiecutter.project_slug.replace("-", "") }}simulation)