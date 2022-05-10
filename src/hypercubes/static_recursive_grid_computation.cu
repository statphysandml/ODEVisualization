#include "../../include/hypercubes/static_recursive_grid_computation.hpp"

StaticRecursiveGridComputation::StaticRecursiveGridComputation(
    const int maximum_recursion_depth,
    const int number_of_cubes_per_gpu_call,
    const int maximum_number_of_gpu_calls
) : DynamicRecursiveGridComputation(number_of_cubes_per_gpu_call, maximum_number_of_gpu_calls),
    grid_computation_wrapper_(GridComputationWrapper(
    number_of_cubes_per_gpu_call_, maximum_recursion_depth))
{}

void StaticRecursiveGridComputation::next(odesolver::DevDatC &vertices)
{
    // Get nodes for the gpu from buffer
    std::tie(node_package_, expected_number_of_cubes_, expected_maximum_depth_) = buffer_.pop_node_package(number_of_cubes_per_gpu_call_);

    if(monitor) {
        std::cout << "\n### Nodes for the qpu: " << node_package_.size() << ", total number of cubes: "
                  << expected_number_of_cubes_ << std::endl;
        buffer_.get_nodes_info(node_package_);
    }
    
    NodesExpander nodesexpander(expected_maximum_depth_, node_package_.size());

    // Fill vectors of length number of nodes
    nodesexpander.extract_node_information(node_package_);
    
    // Fill vectors of length total number of cubes
    grid_computation_wrapper_.expanded_cube_indices_.set_N(expected_number_of_cubes_);
    grid_computation_wrapper_.expanded_depth_per_cube_wrapper_.set_N(expected_number_of_cubes_);
    nodesexpander.expand_node_information(
        node_package_,
        grid_computation_wrapper_.expanded_cube_indices_,
        grid_computation_wrapper_.expanded_depth_per_cube_
    );

    // Compute vertices
    if(vertex_mode_ == CenterVertices)
    {
        vertices.set_N(expected_number_of_cubes_);
        hypercubes_.compute_reference_vertices(vertices, grid_computation_wrapper_, expected_maximum_depth_);
    }
    else if(vertex_mode_ == CubeVertices)
    {
        vertices.set_N(expected_number_of_cubes_ * pow(2, hypercubes_.dim()));
        hypercubes_.compute_vertices(vertices, grid_computation_wrapper_, expected_maximum_depth_);
    }
    else
    {
        vertices.set_N(expected_number_of_cubes_);
        hypercubes_.compute_cube_center_vertices(vertices, grid_computation_wrapper_, expected_maximum_depth_);
    }

    c_++;
}
