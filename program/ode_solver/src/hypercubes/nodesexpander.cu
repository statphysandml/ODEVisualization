#include "../../include/hypercubes/nodesexpander.hpp"


// Fills based on the nodes the given arrays for a later computation of vertices
void NodesExpander::extract_node_information()
{
    // Initialize collected parent cube indices
    auto number_of_nodes_ = number_of_nodes; // to avoid a pass of this within the lambda capture
    thrust::generate(collected_parent_cube_indices.begin(), collected_parent_cube_indices.end(), [number_of_nodes_]() { return dev_vec_int (number_of_nodes_, 0); });

    // Collect counts and values for the expansion
    for(auto node_index = 0; node_index < number_of_nodes; node_index++)
    {
        // Collect cube indices for each dimension
        const std::vector<int> &parent_cube_indices = nodes[node_index]->get_parent_cube_indices();
        // print_range("Parent cube indices", parent_cube_indices.begin(), parent_cube_indices.end());
        int depth = nodes[node_index]->get_depth();
        for(auto depth_index = 0; depth_index < depth; depth_index++)
            collected_parent_cube_indices[depth_index][node_index] = parent_cube_indices[depth_index];

        // Derive depths
        depth_per_node[node_index] = depth;

        // Counter for latter expansion
        number_of_cubes_per_node[node_index] = nodes[node_index]->get_n_cubes();
    }

    // Testing
    /*auto i = 0;
    print_range("Number of cubes per node", number_of_cubes_per_node.begin(), number_of_cubes_per_node.end());
    print_range("Depth per node", depth_per_node.begin(), depth_per_node.end());
    for(auto &elem : collected_parent_cube_indices)
    {
        print_range("Collected parent cube indices in depth " + std::to_string(i), elem.begin(), elem.end());
        i++;
    }*/

    /* Example output
     * Number of cubes per node: 4 8 8 // number_of_cubes_per_node
     * Depth per node: 5 5 5 // depth_per_node
     * Collected parent cube indices in depth 0: 90 90 90 // ...collected_parent_cube_indices...
     * Collected parent cube indices in depth 1: 3 3 3
     * Collected parent cube indices in depth 2: 1 1 1
     * Collected parent cube indices in depth 3: 1 1 1
     * Collected parent cube indices in depth 4: 3 5 7
     */
}

// Cuda code - Expands the node information according to the given number of cubes per node with respect to
// the common parent cube indices -> fills the expanded vectors
GridComputationWrapper NodesExpander::expand_node_information_according_to_number_of_nodes()
{
    GridComputationWrapper grcompwrap(total_number_of_cubes, maximum_depth + 1);

    // Expand parent cube indices
    for(auto depth_index = 0; depth_index < maximum_depth; depth_index++) {
        expand(number_of_cubes_per_node.begin(), number_of_cubes_per_node.end(), collected_parent_cube_indices[depth_index].begin(), grcompwrap.expanded_cube_indices[depth_index].begin());
    }

    // Expand depth
    expand(number_of_cubes_per_node.begin(), number_of_cubes_per_node.end(), depth_per_node.begin(), grcompwrap.expanded_depth_per_cube.begin());

    // Fill expanded cube indices with individual cube indices - fills last row of expanded cube indices
    // -> Adds to each node an individual cube indiex in the deepest recursion step
    // Generate iterators for each depth
    dev_iter_vec_int depth_iterator(maximum_depth + 1);
    for(auto depth_index = 0; depth_index < maximum_depth + 1; depth_index++)
        depth_iterator[depth_index] = new dev_iterator_int(grcompwrap.expanded_cube_indices[depth_index].begin());

    // Fill remaining cube indices
    auto node_index = 0;
    for(auto &node : nodes) {
        int depth = depth_per_node[node_index];//node->get_depth();
        // Copy into correct depth
        *depth_iterator[depth] = thrust::copy(thrust::make_counting_iterator(node->get_internal_start_index()),
                                              thrust::make_counting_iterator(node->get_internal_end_index()),
                                              *depth_iterator[depth]);
        // Increment all other iterators by the amount of vertices for all cubes per node
        for (auto depth_index = 0; depth_index < maximum_depth + 1; depth_index++) {
            if (depth_index != depth)
                *depth_iterator[depth_index] = *depth_iterator[depth_index] + number_of_cubes_per_node[node_index]; // node->get_n_cubes();
        }
        node_index++;
    }

    for(auto depth_index = 0; depth_index < maximum_depth + 1; depth_index++)
        delete depth_iterator[depth_index];


    // Testing
    if(monitor)
    {
        grcompwrap.print_expanded_vectors();
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

    return grcompwrap;
}
