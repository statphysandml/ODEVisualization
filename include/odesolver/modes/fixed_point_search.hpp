#ifndef PROJECT_FIXEDPOINTSEARCH_HPP
#define PROJECT_FIXEDPOINTSEARCH_HPP

#include <sys/file.h>
#include <tuple>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/helper_functions.hpp>
#include <odesolver/util/json_conversions.hpp>
#include <odesolver/collection/buffer.hpp>
#include <odesolver/collection/collection.hpp>
#include <odesolver/collection/leaf.hpp>
#include <odesolver/grid_computation/dynamic_recursive_grid_computation.hpp>
#include <odesolver/grid_computation/grid_computation.hpp>
#include <odesolver/grid_computation/static_recursive_grid_computation.hpp>
#include <odesolver/modes/ode_visualization.hpp>

using json = nlohmann::json;


namespace odesolver {
    namespace modes {
        class FixedPointSearch : public ODEVisualization
        {
        public:
            // From config
            explicit FixedPointSearch(
                const json params,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            // From file
            static FixedPointSearch from_file(
                const std::string rel_config_dir,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            // From parameters
            static FixedPointSearch from_parameters(
                const int maximum_recursion_depth,
                const std::vector<std::vector<int>> n_branches_per_depth,
                const std::vector<std::pair<cudaT, cudaT>> lambda_ranges,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            struct ClusterParameters : public param_helper::params::Parameters
            {
                ClusterParameters(const json params);

                ClusterParameters(
                    const uint maximum_expected_number_of_clusters,
                    const double upper_bound_for_min_distance,
                    const uint maximum_number_of_iterations=1000
                );

                std::string name() const
                {
                    return "cluster";
                }

                const uint maximum_expected_number_of_clusters_;
                const double upper_bound_for_min_distance_;
                const uint maximum_number_of_iterations_;
            };

            static void compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs);

            // Main function

            void find_fixed_points_dynamic_memory();

            void find_fixed_points_preallocated_memory();

            void cluster_solutions_to_fixed_points(const uint maximum_expected_number_of_clusters,
                                const double upper_bound_for_min_distance,
                                const uint maximum_number_of_iterations = 1000);
            void cluster_solutions_to_fixed_points_from_parameters(const FixedPointSearch::ClusterParameters cluster_parameters);
            void cluster_solutions_to_fixed_points_from_file();

            // Getter functions

            std::vector<std::shared_ptr<odesolver::collections::Leaf>> get_solutions();
            odesolver::DevDatC get_fixed_points() const;

            // File interactions

            void write_solutions_to_file(std::string rel_dir) const;
            void load_solutions_from_file(std::string rel_dir);

            void write_fixed_points_to_file(std::string rel_dir) const;
            void load_fixed_points_from_file(std::string rel_dir);

        private:
            uint dim_;
            int maximum_recursion_depth_;

            std::vector<std::vector<int>> n_branches_per_depth_;
            std::vector<std::pair<cudaT, cudaT>> lambda_ranges_;

            std::vector<std::shared_ptr<odesolver::collections::Leaf>> solutions_;
            odesolver::DevDatC fixed_points_;

            // Iterate over collections and generate new collections based on the indices of pot fixed points
            void generate_new_collections_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_fixed_points, const std::vector<odesolver::collections::Collection*> &collections, odesolver::collections::Buffer &buffer);

            thrust::host_vector<int> determine_potential_fixed_points(odesolver::DevDatC& vertex_velocities);
        };

        std::vector<std::vector<double>> load_fixed_points(std::string rel_dir);
    }
}

#endif //PROJECT_FIXEDPOINTSEARCH_HPP
