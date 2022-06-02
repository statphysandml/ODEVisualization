#include "../include/recursive_search_python.hpp"


namespace odesolver {
    namespace pybind {

        void init_recursive_search(py::module &m)
        {
            auto mrecursive_search = m.def_submodule("recursive_search");
            
            py::class_<odesolver::collections::Leaf, std::shared_ptr<odesolver::collections::Leaf>>(mrecursive_search, "Leaf")
                .def(py::init<const std::vector<int>>(), "indices"_a)
                .def_readonly("depth", &odesolver::collections::Leaf::depth_)
                .def_readonly("indices", &odesolver::collections::Leaf::indices_);

            py::class_<odesolver::recursivesearch::RecursiveSearchCriterion, std::shared_ptr<odesolver::recursivesearch::RecursiveSearchCriterion>>(mrecursive_search, "RecursiveSearchCriterion");

            py::class_<odesolver::recursivesearch::FixedPointCriterion, std::shared_ptr<odesolver::recursivesearch::FixedPointCriterion>, odesolver::recursivesearch::RecursiveSearchCriterion>(mrecursive_search, "FixedPointCriterion")
                .def(py::init<uint>(), "dim"_a);
        }
    }
}