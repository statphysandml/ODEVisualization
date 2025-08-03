#ifndef PROGRAM_EVOLUTION_PYTHON_HPP
#define PROGRAM_EVOLUTION_PYTHON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include <iostream>

#include <odesolver/evolution/evolution_observer.hpp>
#include <odesolver/evolution/stepper.hpp>

namespace py = pybind11;
using namespace pybind11::literals;


namespace odesolver {
    namespace pybind {
        void init_evolution(py::module &m);
    }
}

#endif //PROGRAM_EVOLUTION_PYTHON_HPP
