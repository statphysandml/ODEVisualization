#include "../../include/hypercubes/nodesexpander.hpp"

NodesExpander::NodesExpander(
    const int maximum_depth,
    const int number_of_nodes
):
    maximum_depth_(maximum_depth),
    number_of_nodes_(number_of_nodes),
    collected_parent_cube_indices_(thrust::host_vector<dev_vec_int>(maximum_depth)),
    number_of_cubes_per_node_(dev_vec_int(number_of_nodes)),
    depth_per_node_(dev_vec_int(number_of_nodes))
{
    // Initialize collected parent cube indices
    auto n_nodes = number_of_nodes; // to avoid passing "this" within the lambda capture
    thrust::generate(collected_parent_cube_indices_.begin(), collected_parent_cube_indices_.end(), [n_nodes]() { return dev_vec_int(n_nodes, 0); });

    expected_number_of_cubes_ = 0;
    expected_depth_ = 0;
}


void NodesExpander::extract_node_information(const std::vector<Node*> &node_package)
{
    // Collect counts and values for the expansion
    expected_number_of_cubes_ = 0;
    expected_depth_ = 0;

    for(auto node_index = 0; node_index < number_of_nodes_; node_index++)
    {
        // Collect cube indices for each dimension
        const std::vector<int> &parent_cube_indices = node_package[node_index]->get_parent_cube_indices();
        // print_range("Parent cube indices", parent_cube_indices.begin(), parent_cube_indices.end());
        int depth = node_package[node_index]->get_depth();
        for(auto depth_index = 0; depth_index < depth; depth_index++)
            collected_parent_cube_indices_[depth_index][node_index] = parent_cube_indices[depth_index];

        // Derive depths
        depth_per_node_[node_index] = depth;
        if(depth > expected_depth_)
            expected_depth_ = depth;

        // Counter for latter expansion
        number_of_cubes_per_node_[node_index] = node_package[node_index]->get_n_cubes();
    }

    expected_number_of_cubes_ = thrust::reduce(number_of_cubes_per_node_.begin(), number_of_cubes_per_node_.end());

    // Testing
    /*auto i = 0;
    print_range("Number of cubes per node", number_of_cubes_per_node_.begin(), number_of_cubes_per_node_.end());
    print_range("Depth per node", depth_per_node_.begin(), depth_per_node_.end());
    for(auto &elem : collected_parent_cube_indices_)
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


void NodesExpander::expand_node_information(
    const std::vector<Node*> &node_package,
    odesolver::DevDatInt& expanded_cube_indices,
    odesolver::DimensionIteratorInt& expanded_depth_per_cube
)
{
    if(expected_number_of_cubes_ > expanded_cube_indices.n_elems())
    {
        std::cerr << "Number of elements in expanded_cube_indices is too small in comparison to the expected number of cubes." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if(expected_depth_ + 1 > expanded_cube_indices.dim_size())
    {
        std::cerr << "Provided maximum depth in expanded_cube_indices is too small in comparison to the expected maximum recursive depth." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Expand parent cube indices
    for(auto depth_index = 0; depth_index < maximum_depth_; depth_index++) {
        expand(number_of_cubes_per_node_.begin(), number_of_cubes_per_node_.end(), collected_parent_cube_indices_[depth_index].begin(), expanded_cube_indices[depth_index].begin());
    }

    // Expand depth
    expand(number_of_cubes_per_node_.begin(), number_of_cubes_per_node_.end(), depth_per_node_.begin(), expanded_depth_per_cube.begin());

    // [ Fill expanded cube indices with individual cube indices - fills last row of expanded cube indices
    
    // Generate iterators for each depth
    dev_iter_vec_int depth_iterator(maximum_depth_ + 1);
    for(auto depth_index = 0; depth_index < maximum_depth_ + 1; depth_index++)
        depth_iterator[depth_index] = new dev_iterator_int(expanded_cube_indices[depth_index].begin());

    // Fill remaining cube indices
    auto node_index = 0;
    for(auto &node : node_package) {
        int depth = depth_per_node_[node_index]; //Equivalent to node->get_depth();
        // Extract individual cube index and copy into correct depth
        *depth_iterator[depth] = thrust::copy(thrust::make_counting_iterator(node->get_internal_start_index()),
                                              thrust::make_counting_iterator(node->get_internal_end_index()),
                                              *depth_iterator[depth]);
        // Increment all other depth iterators by the amount of vertices according to the number of cubes in the considered node
        for (auto depth_index = 0; depth_index < maximum_depth_ + 1; depth_index++) {
            if (depth_index != depth)
                *depth_iterator[depth_index] = *depth_iterator[depth_index] + number_of_cubes_per_node_[node_index]; // Equivalent to node->get_n_cubes();
        }
        node_index++;
    }

    for(auto depth_index = 0; depth_index < maximum_depth_ + 1; depth_index++)
        delete depth_iterator[depth_index];
    // ]
}
