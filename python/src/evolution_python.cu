#include "../include/evolution_python.hpp"


namespace odesolver {
    namespace pybind {
                
        void init_evolution(py::module &m)
        {
            auto mevolution = m.def_submodule("evolution");

            auto mstepper = mevolution.def_submodule("stepper");

            py::class_<odesolver::evolution::stepper::RungaKutta4, std::shared_ptr<odesolver::evolution::stepper::RungaKutta4>>(mstepper, "RungaKutta4")
                .def(py::init())
                .def("concept", &odesolver::evolution::stepper::RungaKutta4::concept);

            py::class_<odesolver::evolution::stepper::RungaKuttaDopri5, std::shared_ptr<odesolver::evolution::stepper::RungaKuttaDopri5>>(mstepper, "RungaKuttaDopri5")
                .def(py::init())
                .def("concept", &odesolver::evolution::stepper::RungaKuttaDopri5::concept);
            
            py::class_<odesolver::evolution::stepper::ControlledRungaKutta<odesolver::evolution::stepper::RungaKuttaDopri5>, std::shared_ptr<odesolver::evolution::stepper::ControlledRungaKutta<odesolver::evolution::stepper::RungaKuttaDopri5>>>(mstepper, "ControlledRungaKuttaDopri5")
                .def(py::init<double, double>(), "abs_err_tolerance"_a=1.0e-06, "rel_err_tolerance"_a=1.0e-6)
                .def("concept", &odesolver::evolution::stepper::ControlledRungaKutta<odesolver::evolution::stepper::RungaKuttaDopri5>::concept);

            auto mobserver = mevolution.def_submodule("observer");

            py::class_<odesolver::evolution::FlowObserver,
                std::shared_ptr<odesolver::evolution::FlowObserver>>(mobserver, "FlowObserver")
                .def("valid_coordinates_mask", [](odesolver::evolution::FlowObserver &flow_observer) { return flow_observer.valid_coordinates_mask(); })
                .def("n_valid_coordinates", &odesolver::evolution::FlowObserver::n_valid_coordinates)
                .def("valid_coordinates", &odesolver::evolution::FlowObserver::valid_coordinates);
            
            py::class_<odesolver::evolution::DivergentFlow, std::shared_ptr<odesolver::evolution::DivergentFlow>>(mobserver, "DivergentFlow")
                .def(py::init(&odesolver::evolution::DivergentFlow::generate), "flow_equations"_a, "maximum_abs_flow_val"_a=1e10)
                .def("name", &odesolver::evolution::DivergentFlow::name)
                .def("eval", &odesolver::evolution::DivergentFlow::operator());

            py::class_<odesolver::evolution::NoChange, std::shared_ptr<odesolver::evolution::NoChange>>(mobserver, "NoChange")
                .def(py::init(&odesolver::evolution::NoChange::generate), "minimum_change_of_state"_a=std::vector<cudaT>{})
                .def("name", &odesolver::evolution::NoChange::name)
                .def("eval", &odesolver::evolution::NoChange::operator());
            
            py::class_<odesolver::evolution::OutOfRangeCondition, std::shared_ptr<odesolver::evolution::OutOfRangeCondition>>(mobserver, "OutOfRangeCondition")
                .def(py::init(&odesolver::evolution::OutOfRangeCondition::generate), "variable_ranges"_a=std::vector<std::pair<cudaT, cudaT>>{}, "observed_dimension_indices"_a=std::vector<int>{})
                .def("name", &odesolver::evolution::OutOfRangeCondition::name)
                .def("eval", &odesolver::evolution::OutOfRangeCondition::operator());

            py::class_<odesolver::evolution::Intersection, std::shared_ptr<odesolver::evolution::Intersection>>(mobserver, "Intersection")
                .def(py::init(&odesolver::evolution::Intersection::generate))
                .def("name", &odesolver::evolution::Intersection::name)
                .def("eval", &odesolver::evolution::Intersection::operator())
                .def("intersection_counter", &odesolver::evolution::Intersection::intersection_counter)
                .def("detected_intersections", &odesolver::evolution::Intersection::detected_intersections)
                .def("detected_intersection_types", &odesolver::evolution::Intersection::detected_intersection_types);

            py::class_<odesolver::evolution::TrajectoryObserver, std::shared_ptr<odesolver::evolution::TrajectoryObserver>>(mobserver, "TrajectoryObserver")
                .def(py::init(&odesolver::evolution::TrajectoryObserver::generate), "file")
                .def("name", &odesolver::evolution::TrajectoryObserver::name)
                .def("eval", &odesolver::evolution::TrajectoryObserver::operator());
            
            py::class_<odesolver::evolution::EvolutionObserver, std::shared_ptr<odesolver::evolution::EvolutionObserver>>(mobserver, "EvolutionObserver")
                .def(py::init(&odesolver::evolution::EvolutionObserver::generate), "observers"_a)
                .def("name", &odesolver::evolution::EvolutionObserver::name)
                .def("eval", &odesolver::evolution::EvolutionObserver::operator());
        }
    }
}