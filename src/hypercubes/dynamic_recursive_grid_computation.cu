#include "../../include/hypercubes/dynamic_recursive_grid_computation.hpp"

DynamicRecursiveGridComputation::DynamicRecursiveGridComputation(
    const int number_of_cubes_per_gpu_call,
    const int maximum_number_of_gpu_calls) :
    number_of_cubes_per_gpu_call_(number_of_cubes_per_gpu_call),
    maximum_number_of_gpu_calls_(maximum_number_of_gpu_calls)
{}

void DynamicRecursiveGridComputation::initialize(const std::vector<std::vector<int>> n_branches_per_depth,
    const std::vector<std::pair<cudaT, cudaT>> lambda_ranges, VertexMode vertex_mode)
{
    vertex_mode_ = vertex_mode;

    hypercubes_ = HyperCubes(n_branches_per_depth, lambda_ranges);

    std::shared_ptr<Node> root_node_ptr = std::make_shared<Node>(0, compute_internal_end_index(hypercubes_.get_n_branches_per_depth()[0]), std::vector<int>{});
    buffer_ = Buffer(std::move(root_node_ptr));

    c_ = 0;
}

void DynamicRecursiveGridComputation::next(odesolver::DevDatC &vertices)
{
    // Get nodes for the gpu from buffer
    std::tie(node_package_, expected_number_of_cubes_, expected_maximum_depth_) = buffer_.pop_node_package(number_of_cubes_per_gpu_call_);

    if(monitor) {
        std::cout << "\n### Nodes for the qpu: " << node_package_.size() << ", total number of cubes: "
                  << expected_number_of_cubes_ << std::endl;
        buffer_.get_nodes_info(node_package_);
    }

    // Expand nodes
    GridComputationWrapper grid_computation_wrapper(expected_number_of_cubes_, expected_maximum_depth_);
    
    NodesExpander nodesexpander(expected_maximum_depth_, node_package_.size());

    // Fill vectors of length number of nodes
    nodesexpander.extract_node_information(node_package_);
    
    // Fill vectors of length total number of cubes
    nodesexpander.expand_node_information(
        node_package_,
        grid_computation_wrapper.expanded_cube_indices_,
        grid_computation_wrapper.expanded_depth_per_cube_
    );

    // Testing
    if(monitor)
        grid_computation_wrapper.print_expanded_vectors();

    // Compute vertices
    if(vertex_mode_ == CenterVertices)
    {
        vertices = odesolver::DevDatC(hypercubes_.dim() ,expected_number_of_cubes_);
        hypercubes_.compute_reference_vertices(vertices, grid_computation_wrapper);
    }
    else if(vertex_mode_ == CubeVertices)
    {
        vertices = odesolver::DevDatC(hypercubes_.dim(), expected_number_of_cubes_ * pow(2, hypercubes_.dim()));
        hypercubes_.compute_vertices(vertices, grid_computation_wrapper);
    }
    else
    {
        vertices = odesolver::DevDatC(hypercubes_.dim(),expected_number_of_cubes_);
        hypercubes_.compute_cube_center_vertices(vertices, grid_computation_wrapper);
    }

    c_++;
}

bool DynamicRecursiveGridComputation::finished()
{
    return (buffer_.len() == 0) or (c_ > maximum_number_of_gpu_calls_);
}

bool DynamicRecursiveGridComputation::check_status()
{
    if(c_ > maximum_number_of_gpu_calls_)
    {
        std::cout << "Finished because the maximum number of gpu calls has been exceeded." << std::endl;
        return false;
    }
    else
        return true;
}

const std::vector<Node*>& DynamicRecursiveGridComputation::get_node_package()
{
    return node_package_;
}

Buffer& DynamicRecursiveGridComputation::get_buffer()
{
    return buffer_;
}

HyperCubes& DynamicRecursiveGridComputation::get_hypercubes()
{
    return hypercubes_;
}
