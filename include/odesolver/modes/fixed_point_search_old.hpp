#ifndef PROJECT_FIXEDPOINTSEARCH_HPP
#define PROJECT_FIXEDPOINTSEARCH_HPP

#include <sys/file.h>
#include <tuple>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/json_conversions.hpp>
#include <odesolver/collection/buffer.hpp>
#include <odesolver/collection/collection.hpp>
#include <odesolver/collection/leaf.hpp>
#include <odesolver/recursive_search/dynamic_recursive_grid_computation.hpp>
#include <odesolver/grid_computation/grid_computation.hpp>
#include <odesolver/recursive_search/static_recursive_grid_computation.hpp>
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
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            // From parameters
            static FixedPointSearch generate(
                const int maximum_recursion_depth,
                const std::vector<std::vector<int>> n_branches_per_depth,
                const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            // From file
            static FixedPointSearch from_file(
                const std::string rel_config_dir,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            static void compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs);

            // Main function

            void eval(std::string memory_usage="dynamic");

            void evaluate_with_dynamic_memory();

            void evaluate_with_preallocated_memory();

            void project_leaves_on_cube_centers();

            // Getter functions

            const std::vector<std::shared_ptr<odesolver::collections::Leaf>> leaves() const;
            
            const odesolver::DevDatC fixed_points() const;

        private:
            uint dim_;
            int maximum_recursion_depth_;

            std::vector<std::vector<int>> n_branches_per_depth_;
            std::vector<std::pair<cudaT, cudaT>> variable_ranges_;

            std::vector<std::shared_ptr<odesolver::collections::Leaf>> leaves_;
            odesolver::DevDatC fixed_points_;

            // Iterate over collections and generate new collections based on the indices of pot fixed points
            void generate_new_collections_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_fixed_points, const std::vector<odesolver::collections::Collection*> &collections, odesolver::collections::Buffer &buffer);

            thrust::host_vector<int> determine_potential_fixed_points(odesolver::DevDatC& vertex_velocities);
        };

        void write_leaves_to_file(std::string rel_dir, std::vector<odesolver::collections::Leaf*> leaves);

        std::vector<std::shared_ptr<odesolver::collections::Leaf>> load_leaves(std::string rel_dir);
    }
}

#endif //PROJECT_FIXEDPOINTSEARCH_HPP
