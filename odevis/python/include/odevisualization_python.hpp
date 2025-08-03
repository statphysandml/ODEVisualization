#ifndef PROGRAM_ODEVISUALIZATION_PYTHON_HPP
#define PROGRAM_ODEVISUALIZATION_PYTHON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include <iostream>

#include "dev_dat_python.hpp"
#include "evolution_python.hpp"
#include "flow_equations_python.hpp"
#include "modes_python.hpp"
#include "recursive_search_python.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

#endif //PROGRAM_ODEVISUALIZATION_PYTHON_HPP
