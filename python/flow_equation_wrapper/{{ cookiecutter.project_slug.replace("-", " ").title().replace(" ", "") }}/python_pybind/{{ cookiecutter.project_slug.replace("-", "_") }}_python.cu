#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// #include <pybind11_json/pybind11_json.hpp>

#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;


#include <{{ cookiecutter.project_slug.replace("-", "_") }}/{{ cookiecutter.project_slug.replace("-", "_") }}_flow_equation.hpp>
#include <{{ cookiecutter.project_slug.replace("-", "_") }}/{{ cookiecutter.project_slug.replace("-", "_") }}_jacobian_equation.hpp>


void init_{{ cookiecutter.project_slug.replace("-", "_") }}(py::module &m)
{
    py::class_<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations, std::shared_ptr<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations>, odesolver::flowequations::FlowEquationsWrapper>(m, "{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}Flow")
    .def(py::init<cudaT>(), "k"_a=0.0)
    .def("dim", &{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations::get_dim)
    .def_readonly_static("model", &{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations::model_)
    .def_readonly_static("flow_variable", &{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations::explicit_variable_)
    .def_readonly_static("flow_parameters", &{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations::explicit_functions_);

    py::class_<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquations, std::shared_ptr<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquations>, odesolver::flowequations::JacobianEquationsWrapper>(m, "{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}Jacobians")
    .def(py::init<cudaT>(), "k"_a=0.0)
    .def("dim", &{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquations::get_dim)
    .def_readonly_static("model", &{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquations::model_);
}

PYBIND11_MODULE({{ cookiecutter.project_slug.replace("-", "") }}simulation, m)
{
    init_{{ cookiecutter.project_slug.replace("-", "_") }}(m);

    m.doc() = "Python Bindings for {{ cookiecutter.project_name }}";
}
