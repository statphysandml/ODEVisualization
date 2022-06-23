#include <odesolver/grid_computation/grid_computation_wrapper.hpp>


namespace odesolver {
    namespace gridcomputation {
        GridComputationWrapper::GridComputationWrapper(
            const int maximum_number_of_cubes,
            const int maximum_depth,
            const cudaT init_depth_val
        ):
            expanded_element_indices_(devdat::DevDatInt(maximum_depth, maximum_number_of_cubes)),
            expanded_depth_per_element_wrapper_(devdat::DevDatInt(1, maximum_number_of_cubes, init_depth_val)),
            expanded_depth_per_element_(expanded_depth_per_element_wrapper_[0])
        {}

        void GridComputationWrapper::print_expanded_vectors()
        {
            auto i = 0;
            for(auto depth_index = 0; depth_index < expanded_element_indices_.dim_size(); depth_index++) {
                print_range(
                        "Expanded cube indices after filling with individual cube indices in depth " + std::to_string(i),
                        expanded_element_indices_[depth_index].begin(),
                        expanded_element_indices_[depth_index].end());
                i++;
            }
            print_range("Expanded depth per collection", expanded_depth_per_element_.begin(),
                        expanded_depth_per_element_.end());
        }
    }
}
