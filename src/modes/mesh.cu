#include <odesolver/modes/mesh.hpp>


namespace odesolver {
    namespace modes {
        Mesh::Mesh(
            const json params
        ) : Parameters(params),
            n_branches_(get_entry<std::vector<int>>("n_branches")),
            partial_variable_ranges_(devdat::util::json_to_vec_pair<double>(get_entry<json>("variable_ranges"))),
            fixed_variables_(devdat::util::json_to_vec_vec<double>(get_entry<json>("fixed_variables", std::vector<std::vector<cudaT>> {}))),
            partial_ranges_(odesolver::util::PartialRanges(n_branches_, partial_variable_ranges_, fixed_variables_, true))
        {
            // Check consistent definition of n_branches, variable_ranges and fixed_variables
            auto number_of_ones = 0;
            for(auto &n_branch: n_branches_)
            {
                if(n_branch == 1)
                    number_of_ones += 1;
            }
            if(number_of_ones + partial_variable_ranges_.size() != n_branches_.size())
            {
                std::cout << "\nERROR: Inconsistent number of '1'sf in n_branches. The number of ones must be equal to the number of elements for each fixed_variable and the number of non-ones must be equal to variable_ranges.size()." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            for(auto n_branch_index = 0; n_branch_index < n_branches_.size(); n_branch_index++)
            {
                if(n_branches_[n_branch_index] == 1)
                    indices_of_fixed_variables_.push_back(n_branch_index);
            }
        }

        Mesh Mesh::generate(
            const std::vector<int> n_branches,
            const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
            const std::vector<std::vector<cudaT>> fixed_variables
        )
        {
            return Mesh(
                json {{"n_branches", n_branches},
                    {"variable_ranges", variable_ranges},
                    {"fixed_variables", fixed_variables}}
            );
        }

        Mesh Mesh::from_file(
            const std::string rel_config_dir
        )
        {
            return Mesh(
                param_helper::fs::read_parameter_file(
                    param_helper::proj::project_root() + rel_config_dir + "/", "config", false)
            );
        }

        devdat::DevDatC Mesh::eval(int fixed_variable_idx=0)
        {
            /* std::vector<std::pair<cudaT, cudaT>> variable_ranges;
            if(partial_ranges_.size() == 0)
                variable_ranges = partial_variable_ranges_;
            else */
            auto variable_ranges = partial_ranges_[fixed_variable_idx];
            
            odesolver::gridcomputation::GridComputation hypercubes(
                std::vector<std::vector<int>> {n_branches_},
                variable_ranges
            );
            
            // Define collection containing all hypercubes in within variable_ranges
            std::unique_ptr<odesolver::collections::Collection> root_collection_ptr = std::make_unique<odesolver::collections::Collection>(0, odesolver::collections::compute_internal_end_index(n_branches_));
            std::vector<odesolver::collections::Collection*> collections{root_collection_ptr.get()};

            // Expand collection
            auto grid_computation_wrapper = hypercubes.project_collection_package_on_expanded_cube_and_depth_per_cube_indices(collections, root_collection_ptr->size(), 1);
            
            // Compute reference vertices
            return hypercubes.compute_reference_vertices(grid_computation_wrapper);
        }
    }
}