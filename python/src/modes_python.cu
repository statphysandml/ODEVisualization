#include "../include/modes_python.hpp"


namespace odesolver {
    namespace pybind {

        void init_modes(py::module &m)
        {
            auto mmodes = m.def_submodule("modes");
            
            py::class_<odesolver::modes::Evolution, std::shared_ptr<odesolver::modes::Evolution>> evolution(mmodes, "Evolution");
            
            evolution.def(py::init(&odesolver::modes::Evolution::generate), "flow_equations"_a, "jacobian_equations"_a=nullptr);

            init_evolution<odesolver::evolution::stepper::RungaKutta4>(mmodes, evolution);
            init_evolution<odesolver::evolution::stepper::RungaKuttaDopri5>(mmodes, evolution);
            init_evolution<odesolver::evolution::stepper::ControlledRungaKutta<odesolver::evolution::stepper::RungaKuttaDopri5>>(mmodes, evolution);

            init_evolution_observer<odesolver::evolution::stepper::RungaKutta4>(mmodes, evolution);
            init_evolution_observer<odesolver::evolution::stepper::RungaKuttaDopri5>(mmodes, evolution);
            init_evolution_observer<odesolver::evolution::stepper::ControlledRungaKutta<odesolver::evolution::stepper::RungaKuttaDopri5>>(mmodes, evolution);

            py::class_<odesolver::modes::Jacobians, std::shared_ptr<odesolver::modes::Jacobians>>(mmodes, "Jacobians")
                .def(py::init<std::vector<double>, uint>(), "jacobian_elements"_a, "dim"_a)
                .def("from_vec_vec", &odesolver::modes::Jacobians::from_vec_vec)
                .def("from_devdat", &odesolver::modes::Jacobians::from_devdat)
                .def("elements", &odesolver::modes::Jacobians::get_jacobian_elements)
                .def("size", &odesolver::modes::Jacobians::size)
                .def("eval", &odesolver::modes::Jacobians::eval)
                .def("jacobian", &odesolver::modes::Jacobians::get_jacobian)
                .def("eigenvectors", &odesolver::modes::Jacobians::get_eigenvector)
                .def("eigenvalues", &odesolver::modes::Jacobians::get_eigenvalue);
            
            py::class_<odesolver::modes::KMeansClustering, std::shared_ptr<odesolver::modes::KMeansClustering>>(mmodes, "KMeansClustering")
                .def(py::init(&odesolver::modes::KMeansClustering::generate), "maximum_expected_number_of_clusters"_a, "upper_bound_for_min_distance"_a, "maximum_number_of_iterations"_a=1000)
                .def("eval", [](odesolver::modes::KMeansClustering &kmeans_clustering, const devdat::DevDatC &coordinates, int k=-1) {
                    if(k == -1)
                        return kmeans_clustering.eval(coordinates);
                    else
                        return kmeans_clustering.eval(coordinates, k);
                });

            py::class_<odesolver::modes::Mesh, std::shared_ptr<odesolver::modes::Mesh>>(mmodes, "Mesh")
                .def(py::init(&odesolver::modes::Mesh::generate), "n_branches"_a, "variable_ranges"_a, "fixed_variables"_a=std::vector<std::vector<cudaT>>{})
                .def("eval", &odesolver::modes::Mesh::eval)
                .def_readonly("variable_ranges", &odesolver::modes::Mesh::partial_variable_ranges_)
                .def_readonly("fixed_variables", &odesolver::modes::Mesh::fixed_variables_)
                .def_readonly("n_branches", &odesolver::modes::Mesh::n_branches_);
            
            py::class_<odesolver::modes::RecursiveSearch, std::shared_ptr<odesolver::modes::RecursiveSearch>>(mmodes, "RecursiveSearch")
                .def(py::init(&odesolver::modes::RecursiveSearch::generate))
                .def("eval", &odesolver::modes::RecursiveSearch::eval)
                .def("solutions", &odesolver::modes::RecursiveSearch::solutions)
                .def("leaves", &odesolver::modes::RecursiveSearch::leaves);
        }
    }
}