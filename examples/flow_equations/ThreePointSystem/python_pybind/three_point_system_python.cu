#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// #include <pybind11_json/pybind11_json.hpp>

#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;


#include <three_point_system/three_point_system_flow_equation.hpp>
#include <three_point_system/three_point_system_jacobian_equation.hpp>


void init_three_point_system(py::module &m)
{
    py::class_<ThreePointSystemFlowEquations, std::shared_ptr<ThreePointSystemFlowEquations>, flowequations::FlowEquationsWrapper>(m, "ThreePointSystemFlow")
    .def(py::init<cudaT>(), "k"_a=0.0)
    .def("dim", &ThreePointSystemFlowEquations::get_dim)
    .def_readonly_static("model", &ThreePointSystemFlowEquations::model_)
    .def_readonly_static("flow_variable", &ThreePointSystemFlowEquations::explicit_variable_)
    .def_readonly_static("flow_parameters", &ThreePointSystemFlowEquations::explicit_functions_);

    py::class_<ThreePointSystemJacobianEquations, std::shared_ptr<ThreePointSystemJacobianEquations>, flowequations::JacobianEquationsWrapper>(m, "ThreePointSystemJacobians")
    .def(py::init<cudaT>(), "k"_a=0.0)
    .def("dim", &ThreePointSystemJacobianEquations::get_dim)
    .def_readonly_static("model", &ThreePointSystemJacobianEquations::model_);
}

PYBIND11_MODULE(threepointsystemsimulation, m)
{
    init_three_point_system(m);

    m.doc() = "Python Bindings for Three Point System";
}
