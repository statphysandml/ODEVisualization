#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>

#include <odesolver/dev_dat.hpp>


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
            py::class_<DevDatC>(m, "DevDatC")
                .def(py::init<size_t, size_t, cudaT>(),
                    "dim"_a=3, "N"_a=1000, "init_val"_a=0.0)
                .def("dim_size", &DevDatC::dim_size)
                .def("n_elems", &DevDatC::n_elems)
                .def("transpose_device_data", &DevDatC::transpose_device_data)
                .def("print_dim_by_dim", &DevDatC::print_dim_by_dim)
                .def("print_elem_by_elem", &DevDatC::print_elem_by_elem);
        }

        /* void init_measurement_systems(py::module &m)
        {
            py::class_<mcmc::measures::ReadableMeasure>(m, "ReadableMeasure")
                .def(py::init<const std::string>(), "rel_data_dir"_a);
        }

        void init_execution_modes(py::module &m)
        {
            // Equilibrium Time
            py::class_<mcmc::mode::EquilibriumTime>(m, "EquilibriumTime")
                .def(py::init<uint, uint, double, double, std::string>(),
                    "sample_size"_a=100, "number_of_steps"_a=1000, "confidence_range"_a=0.1,
                    "confidence_window"_a=10, "measure"_a="Mean");

            // Correlation Time
            py::class_<mcmc::mode::CorrelationTime>(m, "CorrelationTime")
                .def(py::init<uint, uint, uint, std::string, std::string>(),
                    "minimum_sample_size"_a=100, "maximum_correlation_time"_a=1000, "start_measuring"_a=0,
                    "measure"_a="Mean", "starting_mode"_a="hot")
                .def(py::init<uint, uint, std::string, std::string, std::string>(),
                    "minimum_sample_size"_a=100, "maximum_correlation_time"_a=1000, "equilibrium_time_rel_results_dir"_a,
                    "measure"_a="Mean", "starting_mode"_a="hot");

            // Expectation Value
            py::class_<mcmc::mode::ExpectationValue>(m, "ExpectationValue")
                .def(py::init<uint, uint, uint, std::vector<std::string>, std::vector<std::string>, std::string, std::string, uint>(),
                    "measure_interval"_a=1, "number_of_measurements"_a=1000, "start_measuring"_a=0,
                    "measures"_a=std::vector<std::string>{}, "post_measures"_a=std::vector<std::string>{},
                    "starting_mode"_a="hot", "error_type"_a="statistical", "n_means_bootstrap"_a=0)
                .def(py::init<std::string, uint, uint, std::vector<std::string>, std::vector<std::string>, std::string, std::string, uint>(),
                    "correlation_time_rel_results_dir"_a, "number_of_measurements"_a=1000, "start_measuring"_a=0,
                    "measures"_a=std::vector<std::string>{}, "post_measures"_a=std::vector<std::string>{},
                    "starting_mode"_a="hot", "error_type"_a="statistical", "n_means_bootstrap"_a=0)
                .def(py::init<uint, uint, std::string, std::vector<std::string>, std::vector<std::string>, std::string, std::string, uint>(),
                    "measure_interval"_a=1, "number_of_measurements"_a=1000, "equilibrium_time_rel_results_dir"_a,
                    "measures"_a=std::vector<std::string>{}, "post_measures"_a=std::vector<std::string>{},
                    "starting_mode"_a="hot", "error_type"_a="statistical", "n_means_bootstrap"_a=0)
                .def(py::init<std::string, uint, std::string, std::vector<std::string>, std::vector<std::string>, std::string, std::string, uint>(),
                    "correlation_time_rel_results_dir"_a, "number_of_measurements"_a=1000, "equilibrium_time_rel_results_dir"_a,
                    "measures"_a=std::vector<std::string>{}, "post_measures"_a=std::vector<std::string>{},
                    "starting_mode"_a="hot", "error_type"_a="statistical", "n_means_bootstrap"_a=0);
        } */
    }
}


PYBIND11_MODULE(odevisualization, m)
{
    // odesolver::pybind::init_functions(m);
    odesolver::pybind::init_devdat(m);
    /* mcmc::pybind::init_measurement_systems(m);
    mcmc::pybind::init_execution_modes(m); */

    m.doc() = "Python Bindings for MCMCSimulationLib";
}

