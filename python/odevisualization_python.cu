#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>

#include <odesolver/dev_dat.hpp>
#include <odesolver/flow_equations/flow_equation.hpp>
#include <odesolver/flow_equations/jacobian_equation.hpp>

#include <odesolver/collection/leaf.hpp>

#include <odesolver/modes/recursive_search.hpp>
#include <odesolver/modes/recursive_search_criterion.hpp>
#include <odesolver/modes/fixed_point_criterion.hpp>
#include <odesolver/modes/mesh.hpp>


namespace py = pybind11;
using namespace pybind11::literals;


namespace odesolver {
    namespace pybind {
        /* void init_functions(py::module &m)
        {
            m.def("init_mcmc_python_binding", &mcmc::util::initialize_python, "python_modules_path"_a, "fma_develop"_a=false, "executing_from_python"_a=true);
        } */

        void init_devdat(py::module &m)
        {
            py::class_<DevDatC>(m, "DevDat")
                .def(py::init<size_t, size_t, cudaT>(), "dim"_a=3, "N"_a=1000, "init_val"_a=0.0)
                .def(py::init<std::vector<cudaT>, size_t>(), "data"_a, "dim"_a)
                .def(py::init<std::vector<std::vector<cudaT>>>(), "data"_a)
                .def("dim_size", &DevDatC::dim_size)
                .def("n_elems", &DevDatC::n_elems)
                .def("set_nth_element", &DevDatC::set_nth_element)
                .def("get_nth_element", &DevDatC::get_nth_element)
                .def("fill_dim", [] (DevDatC &devdat, unsigned dim, std::vector<cudaT> data, const int start_idx=0) { 
                    if(int(data.size()) - start_idx > int(devdat.n_elems()))
                    {
                        std::cerr << "error in fill_dim: data.size() - start_idx bigger than n_elems" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    thrust::copy(data.begin(), data.end(), devdat[dim].begin() + start_idx);
                })
                .def("data_in_dim", [] (DevDatC &devdat, unsigned dim, const int start_idx=0, const int end_idx=-1) {
                    int n_elems;
                    if(end_idx == -1)
                        n_elems = devdat.n_elems() - start_idx;
                    else
                        n_elems = end_idx - start_idx;
                    if(n_elems < 0)
                    {
                        std::cerr << "error in data_in_dim: end_idx needs to be bigger or equal to start_idx" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    else if(end_idx > int(devdat.n_elems()))
                    {
                        std::cerr << "error in data_in_dim: end_idx bigger than n_elems()" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    std::vector<cudaT> data(n_elems);
                    thrust::copy(devdat[dim].begin() + start_idx, devdat[dim].begin() + start_idx + n_elems, data.begin());
                    return data;
                })
                .def("write_to_file", &DevDatC::write_to_file);

        }

        void init_flow_equations(py::module &m)
        {
            auto mflow = m.def_submodule("flow");

            py::class_<odesolver::flowequations::FlowEquationsWrapper, std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper>>(mflow, "FlowEquations");

            py::class_<odesolver::flowequations::JacobianEquationsWrapper, std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper>>(mflow, "JacobianEquations");

            mflow.def("compute_flow", [](const odesolver::DevDatC coordinates, std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations) {
                return odesolver::flowequations::compute_flow(coordinates, flow_equations.get());
            });
            
            mflow.def("compute_jacobian_elements", [](const odesolver::DevDatC coordinates, std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobian_equations) {
                return odesolver::flowequations::compute_jacobian_elements(coordinates, jacobian_equations.get());
            });
        }

        void init_recursive_search(py::module &m)
        {
            auto mrecursive_search = m.def_submodule("recursive_search");
            
            py::class_<odesolver::collections::Leaf, std::shared_ptr<odesolver::collections::Leaf>>(mrecursive_search, "Leaf")
                .def(py::init<const std::vector<int>>(), "indices"_a)
                .def_readonly("depth", &odesolver::collections::Leaf::depth_)
                .def_readonly("indices", &odesolver::collections::Leaf::indices_);

            py::class_<odesolver::modes::RecursiveSearchCriterion, std::shared_ptr<odesolver::modes::RecursiveSearchCriterion>>(mrecursive_search, "RecursiveSearchCriterion");

            py::class_<odesolver::modes::FixedPointCriterion, std::shared_ptr<odesolver::modes::FixedPointCriterion>, odesolver::modes::RecursiveSearchCriterion>(mrecursive_search, "FixedPointCriterion")
                .def(py::init<uint>(), "dim"_a);
            
            py::class_<odesolver::modes::RecursiveSearch, std::shared_ptr<odesolver::modes::RecursiveSearch>>(mrecursive_search, "RecursiveSearch")
                .def(py::init(&odesolver::modes::RecursiveSearch::generate))
                .def("eval", &odesolver::modes::RecursiveSearch::eval)
                .def("solutions", &odesolver::modes::RecursiveSearch::solutions)
                .def("leaves", &odesolver::modes::RecursiveSearch::leaves);
        }

        void init_modes(py::module &m)
        {
            auto mmodes = m.def_submodule("modes");

            py::class_<odesolver::modes::Mesh, std::shared_ptr<odesolver::modes::Mesh>>(mmodes, "Mesh")
                .def(py::init(&odesolver::modes::Mesh::generate), "n_branches"_a, "variable_ranges"_a, "fixed_variables"_a=std::vector<std::vector<cudaT>>{})
                .def("eval", &odesolver::modes::Mesh::eval)
                .def_readonly("variable_ranges", &odesolver::modes::Mesh::partial_variable_ranges_)
                .def_readonly("fixed_variables", &odesolver::modes::Mesh::fixed_variables_)
                .def_readonly("n_branches", &odesolver::modes::Mesh::n_branches_);
        }
    }
}


PYBIND11_MODULE(odevisualizationlib, m)
{
    // odesolver::pybind::init_functions(m);
    odesolver::pybind::init_devdat(m);
    odesolver::pybind::init_flow_equations(m);
    odesolver::pybind::init_recursive_search(m);
    odesolver::pybind::init_modes(m);

    m.doc() = "Python Bindings for MCMCSimulationLib";
}

