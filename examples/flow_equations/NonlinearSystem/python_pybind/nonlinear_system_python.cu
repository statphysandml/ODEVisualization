#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// #include <pybind11_json/pybind11_json.hpp>

#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;


#include <nonlinear_system/nonlinear_system_flow_equation.hpp>
#include <nonlinear_system/nonlinear_system_jacobian_equation.hpp>


void init_nonlinear_system(py::module &m)
{
    py::class_<NonlinearSystemFlowEquations, std::shared_ptr<NonlinearSystemFlowEquations>, flowequations::FlowEquationsWrapper>(m, "NonlinearSystemFlow")
    .def(py::init<cudaT>(), "k"_a=0.0)
    .def("dim", &NonlinearSystemFlowEquations::get_dim)
    .def_readonly_static("model", &NonlinearSystemFlowEquations::model_)
    .def_readonly_static("flow_variable", &NonlinearSystemFlowEquations::explicit_variable_)
    .def_readonly_static("flow_parameters", &NonlinearSystemFlowEquations::explicit_functions_);

    py::class_<NonlinearSystemJacobianEquations, std::shared_ptr<NonlinearSystemJacobianEquations>, flowequations::JacobianEquationsWrapper>(m, "NonlinearSystemJacobians")
    .def(py::init<cudaT>(), "k"_a=0.0)
    .def("dim", &NonlinearSystemJacobianEquations::get_dim)
    .def_readonly_static("model", &NonlinearSystemJacobianEquations::model_);
}

PYBIND11_MODULE(nonlinearsystemsimulation, m)
{
    init_nonlinear_system(m);

    m.doc() = "Python Bindings for Nonlinear System";
}
