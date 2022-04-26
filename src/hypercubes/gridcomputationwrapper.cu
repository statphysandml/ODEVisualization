#include "../../include/hypercubes/gridcomputationwrapper.hpp"

GridComputationWrapper::GridComputationWrapper(const int total_number_of_cubes, const int maximum_depth, const cudaT init_depth_val) :
expanded_cube_indices(odesolver::DevDatInt(maximum_depth, total_number_of_cubes)),
expanded_depth_per_cube_wrapper(odesolver::DevDatInt(1, total_number_of_cubes, init_depth_val)),
expanded_depth_per_cube(expanded_depth_per_cube_wrapper[0])
{}

void GridComputationWrapper::print_expanded_vectors()
{
    auto i = 0;
    for (auto depth_index = 0; depth_index < expanded_cube_indices.dim_size(); depth_index++) {
        print_range(
                "Expanded cube indices after filling with individual cube indices in depth " + std::to_string(i),
                expanded_cube_indices[depth_index].begin(),
                expanded_cube_indices[depth_index].end());
        i++;
    }
    print_range("Expanded depth per node", expanded_depth_per_cube.begin(),
                expanded_depth_per_cube.end());
}