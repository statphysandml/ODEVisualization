#include "../../include/hypercubes/gridcomputationwrapper.hpp"

GridComputationWrapper::GridComputationWrapper(
    const int maximum_number_of_cubes,
    const int maximum_depth,
    const cudaT init_depth_val
):
    maximum_number_of_cubes_(maximum_number_of_cubes),
    maximum_depth_(maximum_depth),
    expanded_cube_indices_(odesolver::DevDatInt(maximum_depth, maximum_number_of_cubes)),
    expanded_depth_per_cube_wrapper_(odesolver::DevDatInt(1, maximum_number_of_cubes, init_depth_val)),
    expanded_depth_per_cube_(expanded_depth_per_cube_wrapper_[0])
{}

void GridComputationWrapper::linearise_nodes(const std::vector<Node*> &node_package, int expected_number_of_cubes, int expected_maximum_depth)
{
    if(expected_number_of_cubes != 0)
    {
        expanded_cube_indices_.set_N(expected_number_of_cubes);
        expanded_depth_per_cube_wrapper_.set_N(expected_number_of_cubes);
    }
    auto maximum_depth = maximum_depth_;
    if(expected_maximum_depth != 0)
        maximum_depth = expected_maximum_depth;


    NodesExpander nodesexpander(maximum_depth, node_package.size());

    // Fill vectors of length number of nodes
    nodesexpander.extract_node_information(node_package);
    // Fill vectors of length total number of cubes

    nodesexpander.expand_node_information(
        node_package,
        expanded_cube_indices_,
        expanded_depth_per_cube_
    );

    // Testing
    if(monitor)
    {
        print_expanded_vectors();
    }

    /* Example output
     * Expanded cube indices after filling with individual cube indices in depth 0: 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90
     * Expanded cube indices after filling with individual cube indices in depth 1: 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
     * Expanded cube indices after filling with individual cube indices in depth 2: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     * Expanded cube indices after filling with individual cube indices in depth 3: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     * Expanded cube indices after filling with individual cube indices in depth 4: 3 3 3 3 5 5 5 5 5 5 5 5 7 7 7 7 7 7 7 7
     * Expanded cube indices after filling with individual cube indices in depth 5: 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
     * Expanded depth per node: 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
     */
}


void GridComputationWrapper::print_expanded_vectors()
{
    auto i = 0;
    for (auto depth_index = 0; depth_index < expanded_cube_indices_.dim_size(); depth_index++) {
        print_range(
                "Expanded cube indices after filling with individual cube indices in depth " + std::to_string(i),
                expanded_cube_indices_[depth_index].begin(),
                expanded_cube_indices_[depth_index].end());
        i++;
    }
    print_range("Expanded depth per node", expanded_depth_per_cube_.begin(),
                expanded_depth_per_cube_.end());
}