#ifndef PROGRAM_RECURSIVE_SEARCH_PYTHON_HPP
#define PROGRAM_RECURSIVE_SEARCH_PYTHON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include <iostream>

#include <odesolver/collection/leaf.hpp>
#include <odesolver/recursive_search/recursive_search_criterion.hpp>
#include <odesolver/recursive_search/fixed_point_criterion.hpp>

namespace py = pybind11;
using namespace pybind11::literals;


namespace odesolver {
    namespace pybind {
        void init_recursive_search(py::module &m);
    }
}

#endif //PROGRAM_RECURSIVE_SEARCH_PYTHON_HPP
