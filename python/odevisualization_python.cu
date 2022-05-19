#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>

#include <odesolver/dev_dat.hpp>
#include <odesolver/flow_equations/flow_equation.hpp>
#include <odesolver/flow_equations/jacobian_equation.hpp>

#include <odesolver/modes/fixed_point_search.hpp>


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

        void init_modes(py::module &m)
        {
            auto mmodes = m.def_submodule("modes");

            py::class_<odesolver::modes::FixedPointSearch, std::shared_ptr<odesolver::modes::FixedPointSearch>>(mmodes, "FixedPointSearch")
                .def(py::init(&odesolver::modes::FixedPointSearch::generate))
                .def("eval", &odesolver::modes::FixedPointSearch::eval)
                .def("fixed_points", &odesolver::modes::FixedPointSearch::fixed_points);
                // .def("leaves", &odesolver::modes::FixedPointSearch::leaves);
        }
    }
}


PYBIND11_MODULE(odevisualizationlib, m)
{
    // odesolver::pybind::init_functions(m);
    odesolver::pybind::init_devdat(m);
    odesolver::pybind::init_flow_equations(m);
    odesolver::pybind::init_modes(m);

    m.doc() = "Python Bindings for MCMCSimulationLib";
}

