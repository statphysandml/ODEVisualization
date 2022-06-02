#ifndef PROJECT_VISUALIZATION_HPP
#define PROJECT_VISUALIZATION_HPP

#include <sys/file.h>
#include <tuple>
#include <fstream>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/json_conversions.hpp>
#include <odesolver/util/random.hpp>
#include <odesolver/util/partial_ranges.hpp>
#include <odesolver/collection/buffer.hpp>
#include <odesolver/collection/collection.hpp>
#include <odesolver/collection/leaf.hpp>
#include <odesolver/recursive_search/dynamic_recursive_grid_computation.hpp>
#include <odesolver/grid_computation/grid_computation.hpp>
#include <odesolver/recursive_search/static_recursive_grid_computation.hpp>
#include <odesolver/modes/ode_visualization.hpp>
#include <odesolver/modes/coordinate_operator.hpp>
#include <odesolver/observers/evolution.hpp>
// #include "observers/conditional_intersection_observer.hpp"

using json = nlohmann::json;

/* Examples:
 * - Full variable scan
 * - Scan for unique fixed variables in two dimensions
 * - Scan for fixed variables in two dimensions with one altering value (one fixed variable)
 * - Scan for fixed variables in two dimensions with two altering values (two fixed variables)
 */


namespace odesolver {
    namespace modes {
        class Visualization : public ODEVisualization
        {
        public:
            // From config
            explicit Visualization(
                const json params,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr=nullptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            // From parameters
            static Visualization generate(
                const std::vector<int> n_branches,
                const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
                const std::vector<std::vector<cudaT>> fixed_variables = std::vector< std::vector<cudaT>> {},
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr=nullptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            // From file
            static Visualization from_file(
                const std::string rel_config_dir,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr=nullptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            void append_explicit_points_parameters(
                    std::vector< std::vector<cudaT> > explicit_points=std::vector< std::vector<cudaT> >{},
                    std::string source_root_dir="/data/",
                    bool source_relative_path=true,
                    std::string source_file_dir="");

            struct ComputeVertexVelocitiesParameters : public param_helper::params::Parameters
            {
                ComputeVertexVelocitiesParameters(const json params);

                ComputeVertexVelocitiesParameters(
                        const bool skip_fixed_variables, const bool with_vertices
                );

                std::string name() const {  return "compute_flow";  }

                const bool skip_fixed_variables_;
                const bool with_vertices_;
            };

            struct ComputeSeparatrizesParameters : public param_helper::params::Parameters
            {
                ComputeSeparatrizesParameters(const json params);

                ComputeSeparatrizesParameters(
                        const uint N_per_eigen_dim,
                        const std::vector<double> shift_per_dim
                );

                std::string name() const {  return "compute_separatrizes";  }

                const uint N_per_eigen_dim_;
                const std::vector<double> shift_per_dim_;
            };

            void eval(std::string rel_dir, bool skip_fixed_variables, bool with_vertices);

            // void compute_flow_from_parameters(std::string rel_dir);

            void compute_separatrizes(const std::string rel_dir,
                                    const std::vector<std::pair<cudaT, cudaT>> boundary_variable_ranges,
                                    const std::vector<cudaT> minimum_change_of_state,
                                    const cudaT minimum_delta_t, const cudaT maximum_flow_val,
                                    const std::vector<cudaT> vicinity_distances,
                                    const uint observe_every_nth_step, const uint maximum_total_number_of_steps,
                                    const uint N_per_eigen_dim,
                                    const std::vector<double> shift_per_dim);

            void compute_separatrizes_from_parameters(const std::string rel_dir);

        private:
            const uint dim_;

            std::vector<int> n_branches_;
            std::vector<std::pair<cudaT, cudaT>> partial_variable_ranges_;
            std::vector<std::vector<cudaT>> fixed_variables_;

            std::vector<int> indices_of_fixed_variables_;

            odesolver::DevDatC sample_around_saddle_point(const std::vector<double> coordinate, const std::vector<int> manifold_indices,
                                            const std::vector<std::vector<cudaT>> manifold_eigenvectors, const std::vector<double> shift_per_dim, const uint N_per_eigen_dim);

            odesolver::DevDatC get_initial_values_to_eigenvector(const std::vector<double> saddle_point, std::vector<cudaT> eigenvector, const std::vector<double> shift_per_dim);

            void extract_stable_and_unstable_manifolds(
                    Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType eigenvalue,
                    Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType eigenvector,
                    std::vector<int> &stable_manifold_indices,
                    std::vector<int> &unstable_manifold_indices,
                    std::vector<std::vector<cudaT>> &manifold_eigenvectors
            );

            // ToDo: Check for https://www.boost.org/doc/libs/1_72_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/tutorial/self_expanding_lattices.html to obtain faster results
            void compute_separatrizes_of_manifold(
                    const std::vector<double> saddle_point,
                    const std::vector<int> manifold_indices,
                    const std::vector<std::vector<cudaT>> manifold_eigenvectors,
                    const cudaT delta_t,
                    const std::vector<std::pair<cudaT, cudaT>> boundary_variable_ranges,
                    const std::vector<cudaT> minimum_change_of_state,
                    const cudaT minimum_delta_t, const cudaT maximum_flow_val,
                    const std::vector<cudaT> vicinity_distances,
                    const uint observe_every_nth_step, const uint maximum_total_number_of_steps,
                    const uint N_per_eigen_dim,
                    const std::vector<double> shift_per_dim,
                    std::ofstream &os,
                    std::vector<cudaT> fixed_variables
            );

            std::vector<std::vector<cudaT>> get_fixed_points() const;
        };
    }
}

#endif //PROJECT_VISUALIZATION_HPP
