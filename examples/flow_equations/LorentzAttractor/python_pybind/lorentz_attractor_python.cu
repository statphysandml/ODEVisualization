#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// #include <pybind11_json/pybind11_json.hpp>

#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;


#include <lorentz_attractor/lorentz_attractor_flow_equation.hpp>
#include <lorentz_attractor/lorentz_attractor_jacobian_equation.hpp>


void init_lorentz_attractor(py::module &m)
{
    py::class_<LorentzAttractorFlowEquations, std::shared_ptr<LorentzAttractorFlowEquations>, flowequations::FlowEquationsWrapper>(m, "LorentzAttractorFlow")
    .def(py::init<cudaT>(), "k"_a=0.0)
    .def("dim", &LorentzAttractorFlowEquations::get_dim)
    .def_readonly_static("model", &LorentzAttractorFlowEquations::model_)
    .def_readonly_static("flow_variable", &LorentzAttractorFlowEquations::explicit_variable_)
    .def_readonly_static("flow_parameters", &LorentzAttractorFlowEquations::explicit_functions_);

    py::class_<LorentzAttractorJacobianEquations, std::shared_ptr<LorentzAttractorJacobianEquations>, flowequations::JacobianEquationsWrapper>(m, "LorentzAttractorJacobians")
    .def(py::init<cudaT>(), "k"_a=0.0)
    .def("dim", &LorentzAttractorJacobianEquations::get_dim)
    .def_readonly_static("model", &LorentzAttractorJacobianEquations::model_);
}

PYBIND11_MODULE(lorentzattractorsimulation, m)
{
    init_lorentz_attractor(m);

    m.doc() = "Python Bindings for Lorentz Attractor";
}
