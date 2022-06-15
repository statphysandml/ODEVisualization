#ifndef PROGRAM_MODES_PYTHON_HPP
#define PROGRAM_MODES_PYTHON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include <iostream>

#include <odesolver/evolution/stepper.hpp>

#include <odesolver/modes/evolution.hpp>
#include <odesolver/modes/jacobians.hpp>
#include <odesolver/modes/kmeans_clustering.hpp>
#include <odesolver/modes/mesh.hpp>
#include <odesolver/modes/recursive_search.hpp>

namespace py = pybind11;
using namespace pybind11::literals;


namespace odesolver {
    namespace pybind {

        template<typename StepperClass>
        void init_evolution(py::module &m, py::class_<odesolver::modes::Evolution, std::shared_ptr<odesolver::modes::Evolution>> &evolution)
        {   
            evolution.def("evolve_const", &odesolver::modes::Evolution::evolve_const<StepperClass>, "stepper"_a, "coordinates"_a, "start_t"_a, "end_t"_a, "delta_t"_a)
            .def("evolve_n_steps", &odesolver::modes::Evolution::evolve_n_steps<StepperClass>, "stepper"_a, "coordinates"_a, "start_t"_a, "delta_t"_a, "n"_a);
        }

        template<typename StepperClass, typename Observer>
        void init_evolution_observer(py::module &m, py::class_<odesolver::modes::Evolution, std::shared_ptr<odesolver::modes::Evolution>> &evolution)
        {
            evolution.def("evolve_const", &odesolver::modes::Evolution::evolve_const<StepperClass, Observer>, "stepper"_a, "coordinates"_a, "start_t"_a, "end_t"_a, "delta_t"_a, "observer"_a, "equidistant_time_observations"_a=true, "observe_every_ith_time_step"_a=1)
            .def("evolve_n_steps", &odesolver::modes::Evolution::evolve_n_steps<StepperClass, Observer>, "stepper"_a, "coordinates"_a, "start_t"_a, "delta_t"_a, "n"_a, "observer"_a, "equidistant_time_observations"_a=true, "observe_every_ith_time_step"_a=1);
        }

        template<typename StepperClass>
        void init_evolution_observer(py::module &m, py::class_<odesolver::modes::Evolution, std::shared_ptr<odesolver::modes::Evolution>> &evolution)
        {
            init_evolution_observer<StepperClass, odesolver::evolution::DivergentFlow>(m, evolution);
            init_evolution_observer<StepperClass, odesolver::evolution::NoChange>(m, evolution);
            init_evolution_observer<StepperClass, odesolver::evolution::OutOfRangeCondition>(m, evolution);
            init_evolution_observer<StepperClass, odesolver::evolution::Intersection>(m, evolution);
            // init_evolution_observer<StepperClass, 
            // odesolver::evolution::TrajectoryObserver>(m);
            init_evolution_observer<StepperClass, odesolver::evolution::EvolutionObserver>(m, evolution);
        }

        void init_modes(py::module &m);
    }
}

#endif //PROGRAM_MODES_PYTHON_HPP
