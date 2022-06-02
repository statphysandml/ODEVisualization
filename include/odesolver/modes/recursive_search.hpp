#ifndef PROJECT_RECURSIVESEARCH_HPP
#define PROJECT_RECURSIVESEARCH_HPP

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
#include <odesolver/recursive_search/recursive_search_criterion.hpp>
#include <odesolver/recursive_search/fixed_point_criterion.hpp>
#include <odesolver/modes/ode_visualization.hpp>

using json = nlohmann::json;


namespace odesolver {
    namespace modes {
        class RecursiveSearch : public ODEVisualization
        {
        public:
            // From config
            explicit RecursiveSearch(
                const json params,
                std::shared_ptr<odesolver::recursivesearch::RecursiveSearchCriterion> criterion_ptr,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr
            );

            // From parameters
            static RecursiveSearch generate(
                const int maximum_recursion_depth,
                const std::vector<std::vector<int>> n_branches_per_depth,
                const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
                std::shared_ptr<odesolver::recursivesearch::RecursiveSearchCriterion> criterion_ptr,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const int number_of_cubes_per_gpu_call = 20000,
                const int maximum_number_of_gpu_calls = 1000
            );

            // From file
            static RecursiveSearch from_file(
                const std::string rel_config_dir,
                std::shared_ptr<odesolver::recursivesearch::RecursiveSearchCriterion> criterion_ptr,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr
            );

            // Main function

            void eval(std::string memory_usage="dynamic");

            void evaluate_with_dynamic_memory();

            void evaluate_with_preallocated_memory();

            // -> ToDo: Move?
            void project_leaves_on_cube_centers();

            // Getter functions

            const std::vector<std::shared_ptr<odesolver::collections::Leaf>> leaves() const;
            
            const odesolver::DevDatC solutions() const;

        private:
            uint dim_;
            int maximum_recursion_depth_;
            int number_of_cubes_per_gpu_call_;
            int maximum_number_of_gpu_calls_;

            std::vector<std::vector<int>> n_branches_per_depth_;
            std::vector<std::pair<cudaT, cudaT>> variable_ranges_;

            std::vector<std::shared_ptr<odesolver::collections::Leaf>> leaves_;
            odesolver::DevDatC solutions_;

            std::shared_ptr<odesolver::recursivesearch::RecursiveSearchCriterion> criterion_ptr_;

            // Iterate over collections and generate new collections based on the indices of pot fixed points
            void generate_new_collections_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_solutions, const std::vector<odesolver::collections::Collection*> &collections, odesolver::collections::Buffer &buffer);
        };
    }
}

#endif //PROJECT_RECURSIVESEARCH_HPP
