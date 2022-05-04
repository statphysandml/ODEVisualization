#include "../include/odesolver/fixed_point_search.hpp"

// FixedPointSearch Constructors

FixedPointSearch::FixedPointSearch(
    const json params,
    std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr,
    std::shared_ptr<JacobianEquationWrapper> jacobians_ptr,
    const std::string computation_parameters_path
) : ODEVisualisation(params, flow_equations_ptr, jacobians_ptr, computation_parameters_path),
    dim_(get_entry<json>("flow_equation")["dim"].get<cudaT>()),
    maximum_recursion_depth_(get_entry<int>("maximum_recursion_depth"))
{
    hypercubes_ = HyperCubes(
        json_to_vec_vec<int>(get_entry<json>("n_branches_per_depth")), json_to_vec_pair<double>(get_entry<json>("lambda_ranges"))
    );

    std::shared_ptr<Node> root_node_ptr = std::make_shared<Node>(0, compute_internal_end_index(hypercubes_.get_n_branches_per_depth()[0]), std::vector< int >{});

    buffer_ = Buffer(std::move(root_node_ptr));

    // Tests
    if (hypercubes_.get_n_branches_per_depth().size() < maximum_recursion_depth_)
    {
        std::cout << "\nERROR: Maximum recursion depth " << maximum_recursion_depth_
                  << " is higher than the available number of branches per depth " << hypercubes_.get_n_branches_per_depth().size()
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (const auto &n_branches : hypercubes_.get_n_branches_per_depth())
    {
        if (n_branches.size() != dim_) {
            std::cout << "\nERROR: Number of branches per depth " << n_branches.size() << " do not coincide with dimension " << dim_ <<  std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    if(hypercubes_.get_lambda_ranges().size() != dim_)
    {
        std::cout << "\nERROR: Number of lambda ranges " << hypercubes_.get_lambda_ranges().size() << " do not coincide with dimension" << dim_ << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if(flow_equations_ptr_->get_dim() != dim_)
    {
        std::cout << "\nERROR: Dimensions and number of flow equation do not coincide" << dim_ << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

FixedPointSearch FixedPointSearch::from_file(
    const std::string rel_config_dir,
    std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr,
    std::shared_ptr<JacobianEquationWrapper> jacobians_ptr,
    const std::string computation_parameters_path
)
{
    return FixedPointSearch(
        param_helper::fs::read_parameter_file(
            param_helper::proj::project_root() + rel_config_dir + "/", "config", false),
        flow_equations_ptr,
        jacobians_ptr,
        computation_parameters_path
    );
}

FixedPointSearch FixedPointSearch::from_parameters(
        const int maximum_recursion_depth,
        const std::vector< std::vector<int> > n_branches_per_depth,
        const std::vector <std::pair<cudaT, cudaT> > lambda_ranges,
        std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr,
        std::shared_ptr<JacobianEquationWrapper> jacobians_ptr,
        const std::string computation_parameters_path
)
{
    return FixedPointSearch(
        json {{"maximum_recursion_depth", maximum_recursion_depth},
              {"n_branches_per_depth", n_branches_per_depth},
              {"lambda_ranges", lambda_ranges}},
        flow_equations_ptr,
        jacobians_ptr,
        computation_parameters_path
    );
}


FixedPointSearch::ClusterParameters::ClusterParameters(const json params) : Parameters(params),
                                        maximum_expected_number_of_clusters_(get_entry<uint>("maximum_expected_number_of_clusters")),
                                        upper_bound_for_min_distance_(get_entry<double>("upper_bound_for_min_distance")),
                                        maximum_number_of_iterations_(get_entry<uint>("maximum_number_of_iterations"))
{}

FixedPointSearch::ClusterParameters::ClusterParameters(
        const uint maximum_expected_number_of_clusters,
        const double upper_bound_for_min_distance,
        const uint maximum_number_of_iterations
) : ClusterParameters(
        json {{"maximum_expected_number_of_clusters", maximum_expected_number_of_clusters},
              {"upper_bound_for_min_distance", upper_bound_for_min_distance},
              {"maximum_number_of_iterations", maximum_number_of_iterations}}
)
{}

// Main functions

void FixedPointSearch::cluster_solutions_to_fixed_points(const uint maximum_expected_number_of_clusters,
        const double upper_bound_for_min_distance,
        const uint maximum_number_of_iterations)
{
    // Compute vertices of solutions
    std::vector<Leaf*> solutions;
    std::transform(solutions_.begin(), solutions_.end(), std::back_inserter(solutions), [](const std::shared_ptr<Leaf>& leaf_ptr) {
        return leaf_ptr.get();
    });
    GridComputationWrapper grcompwrap = hypercubes_.project_leaves_on_expanded_cube_and_depth_per_cube_indices(solutions);
    
    // Get center vertices
    const odesolver::DevDatC potential_fixed_points = hypercubes_.compute_cube_center_vertices(grcompwrap);

    // Cluster center vertices
    fixed_points_ = cluster_device_data(
            maximum_expected_number_of_clusters,
            upper_bound_for_min_distance,
            potential_fixed_points,
            maximum_number_of_iterations
            );
}

void FixedPointSearch::cluster_solutions_to_fixed_points_from_parameters(const FixedPointSearch::ClusterParameters cluster_parameters)
{
    cluster_solutions_to_fixed_points(
            cluster_parameters.maximum_expected_number_of_clusters_,
            cluster_parameters.upper_bound_for_min_distance_,
            cluster_parameters.maximum_number_of_iterations_);
}

void FixedPointSearch::cluster_solutions_to_fixed_points_from_file()
{
    auto cluster_params = get_entry<json>("cluster");
    auto cluster_parameters = FixedPointSearch::ClusterParameters(cluster_params);
    cluster_solutions_to_fixed_points_from_parameters(cluster_parameters);
}

std::vector<std::shared_ptr<Leaf>> FixedPointSearch::get_solutions()
{
    return solutions_;
}

odesolver::DevDatC FixedPointSearch::get_fixed_points() const
{
    return fixed_points_;
}

void FixedPointSearch::write_solutions_to_file(std::string rel_dir) const
{
    json j;
    for(auto &sol: solutions_)
        j.push_back(sol->to_json());
    param_helper::fs::write_parameter_file(json {{"number_of_solutions", solutions_.size()}, {"solutions", j}}, param_helper::proj::project_root() + rel_dir + "/", "solutions", false);
}

void FixedPointSearch::load_solutions_from_file(std::string rel_dir)
{
    solutions_.clear();

    json j = param_helper::fs::read_parameter_file(param_helper::proj::project_root() + rel_dir + "/", "solutions", false);
    solutions_.reserve(j["number_of_solutions"].get<int>());
    for(auto &sol: j["solutions"])
        solutions_.push_back(std::make_shared<Leaf>(sol["cube_indices"].get<std::vector<int>>()));
    std::cout << "solutions loaded" << std::endl;
}

void FixedPointSearch::write_fixed_points_to_file(std::string rel_dir) const
{
    auto transposed_fixed_points = fixed_points_.transpose_device_data();
    param_helper::fs::write_parameter_file(
        json {
            {"number_of_fixed_points", transposed_fixed_points.size()},
            {"fixed_points", vec_vec_to_json(transposed_fixed_points)}
        },
        param_helper::proj::project_root() + "/" + rel_dir + "/", "fixed_points",
        false
    );
}

void FixedPointSearch::load_fixed_points_from_file(std::string rel_dir)
{
    auto fixed_points = load_fixed_points(rel_dir);
    fixed_points_ = odesolver::DevDatC(fixed_points);
    std::cout << "fixed points loaded" << std::endl;
}

// Iterate over nodes and generate new nodes based on the indices of pot fixed points
void FixedPointSearch::generate_new_nodes_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_fixed_points, const std::vector<Node*> &nodes)
{
    int n_new_nodes = 0;
    int n_new_leaves = 0;

    // No potential fixed points have been found
    if(host_indices_of_pot_fixed_points.size() > 0)
    {
        // Initial conditions
        auto pot_fixed_point_iterator = host_indices_of_pot_fixed_points.begin();
        int index_offset = 0;

        // Iterate over nodes
        for(const auto &node : nodes)
        {
            // Get first potential fix point -> is defined with respect to 0
            int index_of_pot_fixed_point = *pot_fixed_point_iterator - index_offset; // (-1 to undo offset) -> not used anymore (why initially used??)

            // Fix points have been found in node
            if(index_of_pot_fixed_point < node->get_n_cubes())
            {
                // Inspect fixed points
                while(index_of_pot_fixed_point < node->get_n_cubes() and pot_fixed_point_iterator != host_indices_of_pot_fixed_points.end())
                {
                    // Compute parent node indices
                    std::vector<int> parent_cube_indices(node->get_parent_cube_indices());
                    parent_cube_indices.push_back(index_of_pot_fixed_point + node->get_internal_start_index());

                    // Generate new nodes
                    if(node->get_depth() + 1 < maximum_recursion_depth_) {
                        buffer_.append_node(0, compute_internal_end_index(hypercubes_.get_n_branches_per_depth()[node->get_depth() + 1]), parent_cube_indices);
                        n_new_nodes++;
                    }
                    else // Found solution -> Generate new leaf
                    {
                        solutions_.push_back(std::make_shared<Leaf>(parent_cube_indices));
                        n_new_leaves++;
                    }
                    // Update
                    pot_fixed_point_iterator++;
                    index_of_pot_fixed_point = *pot_fixed_point_iterator - index_offset; // (-1 to undo offset) -> not used anymore (why initially used??)
                }
            }

            // Update index offset
            index_offset += node->get_n_cubes();
        }

        assert(host_indices_of_pot_fixed_points.size() ==  n_new_nodes + n_new_leaves && "Number of new nodes and number of potential fixed points do not coincide");
    }

    // buffer_.add_nodes(new_nodes);
    // solutions_.insert(solutions_.end(), new_leaves.begin(), new_leaves.end());

    /* if(monitor) {
        std::cout << "\n### New nodes" << std::endl;
        buffer_.get_nodes_info(new_nodes);
    } */
}

void FixedPointSearch::find_fixed_points_dynamic_memory()
{
    auto c = 0;

    std::vector<Node*> node_package;
    int expected_number_of_cubes = 0;
    int maximum_depth = 0;

    while(c < computation_parameters_.maximum_number_of_gpu_calls_ and buffer_.len() > 0)
    {
        std::cout << "\n\n######### New computation round: " << c <<  " #########" << std::endl;
    
        // Get nodes for the gpu from buffer
        std::tie(node_package, expected_number_of_cubes, maximum_depth) = buffer_.pop_node_package(computation_parameters_.number_of_cubes_per_gpu_call_);
    
        if(monitor) {
            std::cout << "\n### Nodes for the qpu: " << node_package.size() << ", total number of cubes: "
                      << expected_number_of_cubes << std::endl;
            buffer_.get_nodes_info(node_package);
        }
    
        GridComputationWrapper grid_computation_wrapper(expected_number_of_cubes, maximum_depth + 1);
    
        // Use helper class to perform gpu tasks on nodes
        grid_computation_wrapper.linearise_nodes(node_package);
    
        // Compute the actual vertices by first expanding each cube according to the number of vertices to
        // a vector of reference vertices of length expected_number_of_cubes*dim and then computing the indices
        auto vertices = hypercubes_.compute_vertices(grid_computation_wrapper);
    
        // hypercubes.test_projection();
    
        // Compute vertex velocities
        auto vertex_velocities = compute_vertex_velocities(vertices, flow_equations_ptr_.get());
        // hypercubes.determine_vertex_velocities(flow_equations_ptr_));
    
        // Determine potential fixed points
        thrust::host_vector<int> host_indices_of_pot_fixed_points = hypercubes_.determine_potential_fixed_points(vertex_velocities);
    
        // Generate new nodes and derive solutions based on nodes and indices of potential fixed points
        generate_new_nodes_and_leaves(host_indices_of_pot_fixed_points, node_package);

        c++;
    }
}

// ToDo: Recheck recusrion depth.... by rerunning the simple commented test functions
// ToDo:
// - Take into account maximum depth at ALL stages
// - Fix total_number_of_cubes dependence in hypercubes
// - Take into account finite number of actual vertices in the computation of vertex_velocities (and maybe also in determine potential fixed points!)
void FixedPointSearch::find_fixed_points_preallocated_memory()
{
    // Initialize grid computation wrapper
    GridComputationWrapper grid_computation_wrapper(computation_parameters_.number_of_cubes_per_gpu_call_, maximum_recursion_depth_);

    // Initialize vertices
    odesolver::DevDatC vertices(dim_, computation_parameters_.number_of_cubes_per_gpu_call_ * pow(2, dim_));

    // Initialize vertex velocities
    odesolver::DevDatC vertex_velocities(dim_, computation_parameters_.number_of_cubes_per_gpu_call_ * pow(2, dim_));

    auto c = 0;
    while(c < computation_parameters_.maximum_number_of_gpu_calls_ and buffer_.len() > 0)
    {
        std::cout << "\n\n######### New computation round: " << c <<  " #########" << std::endl;
        
        std::vector<Node*> node_package;
        int expected_number_of_cubes = 0;
        int maximum_depth = 0;
    
        // Get nodes for the gpu from buffer
        std::tie(node_package, expected_number_of_cubes, maximum_depth) = buffer_.pop_node_package(computation_parameters_.number_of_cubes_per_gpu_call_);
    
        if(monitor) {
            std::cout << "\n### Nodes for the qpu: " << node_package.size() << ", total number of cubes: "
                      << expected_number_of_cubes << std::endl;
            buffer_.get_nodes_info(node_package);
        }
    
        // Use helper class to perform gpu tasks on nodes
        // Note that in this case no preallocation is performed due to the small number of generally expected nodes.
        
        // ToDo: Expected depth not taken into account!
        grid_computation_wrapper.linearise_nodes(node_package);
    
        // Compute the actual vertices by first expanding each cube according to the number of vertices to
        // a vector of reference vertices of length expected_number_of_cubes*dim and then computing the indices
        hypercubes_.compute_vertices(vertices, grid_computation_wrapper, expected_number_of_cubes);
    
        // hypercubes.test_projection();
    
        // Compute vertex velocities
        compute_vertex_velocities(vertices, vertex_velocities, flow_equations_ptr_.get());
        // hypercubes.determine_vertex_velocities(flow_equations_ptr_));
    
        // Determine potential fixed points
        thrust::host_vector<int> host_indices_of_pot_fixed_points = hypercubes_.determine_potential_fixed_points(vertex_velocities);
    
        // Generate new nodes and derive solutions based on nodes and indices of potential fixed points
        generate_new_nodes_and_leaves(host_indices_of_pot_fixed_points, node_package);

        c++;
    }
}

std::vector<std::vector<double>> load_fixed_points(std::string rel_dir)
{
    json j = param_helper::fs::read_parameter_file(param_helper::proj::project_root() + rel_dir + "/", "fixed_points", false);
    return json_to_vec_vec(j["fixed_points"]);
}