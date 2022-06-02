#ifndef PROGRAM_FLOW_EQUATIONS_PYTHON_HPP
#define PROGRAM_FLOW_EQUATIONS_PYTHON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include <iostream>

#include <odesolver/dev_dat.hpp>
#include <odesolver/flow_equations/flow_equation.hpp>
#include <odesolver/flow_equations/jacobian_equation.hpp>

namespace py = pybind11;
using namespace pybind11::literals;


namespace odesolver {
    namespace pybind {
        void init_flow_equations(py::module &m);
    }
}

#endif //PROGRAM_FLOW_EQUATIONS_PYTHON_HPP
