#include "../include/visualization.hpp"


struct finalize_sample_around_saddle_point
{
    finalize_sample_around_saddle_point(const cudaT coordinate_val_, const cudaT shift_) :
        coordinate_val(coordinate_val_), shift(shift_)
    {}
    __host__ __device__
    cudaT operator()(const cudaT & sampled_val)
    {

        return coordinate_val +  shift * sampled_val;
    }

    const cudaT coordinate_val;
    const cudaT shift;
};


struct normalize_by_square_root
{
    __host__ __device__
    cudaT operator() (const cudaT &val1, const cudaT &val2) {
        return val1 / std::sqrt(val2);
    }
};


struct sum_square
{
    __host__ __device__
    cudaT operator() (const cudaT &val1, const cudaT &val2) {
        return val1 + val2 * val2;
    }
};

struct sum_manifold_eigenvector
{
    sum_manifold_eigenvector(const cudaT vector_elem_) : vector_elem(vector_elem_)
    {}

    __host__ __device__
    cudaT operator() (const cudaT &previous_val, const cudaT random_number) {
        return previous_val + random_number * vector_elem;
    }

    const cudaT vector_elem;
};


VisualizationParameters::VisualizationParameters(const json params_, const PathParameters path_parameters_) : FRGVisualizationParameters(params_, path_parameters_),
                                                                       dim(get_value_by_key<int>("dim")),
                                                                       k(get_value_by_key<cudaT>("k"))
{
    auto n_branches_ = get_value_by_key<json>("n_branches");
    auto partial_lambda_ranges_ = get_value_by_key<json>("partial_lambda_ranges");
    auto fix_lambdas_ = get_value_by_key<json>("fix_lambdas");

    n_branches = n_branches_.get< std::vector<int> >();
    std::transform(partial_lambda_ranges_.begin(), partial_lambda_ranges_.end(), std::back_inserter(partial_lambda_ranges),
                   [] (json &dat) { return dat.get< std::pair<cudaT, cudaT> >(); });
    std::transform(fix_lambdas_.begin(), fix_lambdas_.end(), std::back_inserter(fix_lambdas),
                   [] (json &dat) { return dat.get< std::vector<cudaT> >(); });

    std::string theory = path_parameters.theory;
    flow_equations = FlowEquationsWrapper::make_flow_equation(theory);
}


VisualizationParameters::VisualizationParameters(const std::string theory,
                                                 const std::string mode_type,
                                                 const std::string results_dir,
                                                 const std::string root_dir,
                                                 const bool relative_path) : VisualizationParameters(
        Parameters::read_parameter_file(
                root_dir + "/" + theory + "/" + results_dir + "/", "config", relative_path),
        PathParameters(theory, mode_type, root_dir, relative_path))
{}


VisualizationParameters::VisualizationParameters(
        const std::string theory,
        const std::vector<int> n_branches_,
        const std::vector <std::pair<cudaT, cudaT> > lambda_ranges_,
        const std::vector <std::vector <cudaT> > fix_lambdas_,
        const std::string mode_,
        const std::string root_dir,
        const bool relative_path
) : VisualizationParameters(
        json {{"n_branches", n_branches_},
              {"partial_lambda_ranges", lambda_ranges_},
              {"fix_lambdas", fix_lambdas_},
              {"mode", mode_}},
        PathParameters(theory, mode_, root_dir, relative_path)
)
{}


void VisualizationParameters::append_explicit_points_parameters(
        std::vector< std::vector<cudaT> > explicit_points,
        std::string source_root_dir,
        bool source_relative_path,
        std::string source_file_dir)
{
    // Will be imported from file
    if(explicit_points.size() == 0)
    {
        // ToDo: Correct?? -> No
        std::string path = get_absolute_path(path_parameters.get_base_path() + source_file_dir + "/fixed_points.json", source_relative_path);
        params["explicit_points"]["from_file"] = path;
    }
        // Explicit
    else
    {
        json j;
        for(auto &fixed_point : explicit_points)
            j.push_back(fixed_point);
        params["explicit_points"] = json {{"number_of_explicit_points", explicit_points.size()}, {"explicit_points", j}};
    }
}


VisualizationParameters::ComputeVertexVelocitiesParameters::ComputeVertexVelocitiesParameters(const json params_) : Parameters(params_),
                                                        skip_fix_lambdas(get_value_by_key<bool>("skip_fix_lambdas")),
                                                        with_vertices(get_value_by_key<bool>("with_vertices"))
{}

VisualizationParameters::ComputeVertexVelocitiesParameters::ComputeVertexVelocitiesParameters(
        const bool skip_fix_lambdas_, const bool with_vertices_
) : ComputeVertexVelocitiesParameters(
        json {{"skip_fix_lambdas", skip_fix_lambdas_},
              {"with_vertices", with_vertices_}}
)
{}


VisualizationParameters::ComputeSeparatrizesParameters::ComputeSeparatrizesParameters(const json params_) : Parameters(params_),
                                                    N_per_eigen_dim(get_value_by_key<uint>("N_per_eigen_dim")),
                                                    shift_per_dim(get_value_by_key<std::vector<double>>("shift_per_dim"))
{}

VisualizationParameters::ComputeSeparatrizesParameters::ComputeSeparatrizesParameters(
        const uint N_per_eigen_dim_,
        const std::vector<double> shift_per_dim_
) : ComputeSeparatrizesParameters(
        json {{"N_per_eigen_dim", N_per_eigen_dim_},
              {"shift_per_dim", shift_per_dim_}}
)
{}


std::vector< std::vector<cudaT> > VisualizationParameters::get_fixed_points() const
{
    auto fixed_points_ = get_value_by_key<json>("fixed_points")["fixed_points"];
    std::vector < std::vector<cudaT> > fixed_points;
    std::transform(fixed_points_.begin(), fixed_points_.end(), std::back_inserter(fixed_points),
                   [] (json &dat) { return dat.get< std::vector<cudaT> >(); });
    return fixed_points;
}


Visualization::Visualization(const VisualizationParameters &vp_) : vp(vp_)
{
    std::cout << vp.dim << std::endl;        // Tests
    if (vp.n_branches.size() != vp.dim) {
        std::cout << "\nERROR: Number of branches per depth " << vp.n_branches.size() << " do not coincide with dimension " << vp.dim <<  std::endl;
        std::exit(EXIT_FAILURE);
    }

    if(vp.partial_lambda_ranges.size() + vp.fix_lambdas.size() != vp.dim)
    {
        std::cout << "\nERROR: Number of lambda ranges and fix lambdas " << vp.partial_lambda_ranges.size() << ", " << vp.fix_lambdas.size() << " do not coincide with dimension " << vp.dim << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if(vp.flow_equations->get_dim() != vp.dim)
    {
        std::cout << "\nERROR: Dimensions and number of flow equation do not coincide" << vp.dim << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Check consistent definition of n_branches, lambda_ranges and fix_lambdas
    auto number_of_ones = 0;
    for(auto &n_branch: vp.n_branches)
    {
        if(n_branch == 1)
            number_of_ones += 1;
    }
    if(number_of_ones != vp.fix_lambdas.size())
    {
        std::cout << "\nERROR: Inconsistent definition of n_branches and fix_lambdas -> cannot expand lambda range for n_branch=1" << vp.dim << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for(auto n_branch_index = 0; n_branch_index < vp.n_branches.size(); n_branch_index++)
    {
        if(vp.n_branches[n_branch_index] == 1)
            indices_of_fixed_lambdas.push_back(n_branch_index);
    }
}

void Visualization::compute_vertex_velocities(std::string dir, bool skip_fix_lambdas, bool with_vertices)
{
    std::ofstream os, os_vertices;
    std::string path = vp.get_absolute_path(vp.path_parameters.get_base_path() + "/" + dir + "/", vp.path_parameters.relative_path);
    os.open(path + "velocities" + ".dat");

    if(with_vertices)
        os_vertices.open(path + "vertices" + ".dat");

    std::vector<int> skip_iterators_in_dimensions {};
    for(auto i = 0; i < vp.n_branches.size(); i++)
    {
        if(vp.n_branches[i] == 1)
            skip_iterators_in_dimensions.push_back(i);
    }


    LambdaRangeGenerator lambda_range_generator(vp.n_branches, vp.partial_lambda_ranges, vp.fix_lambdas);
    auto c = 0;
    while(!lambda_range_generator.finished())
    {
        auto lambda_ranges = lambda_range_generator.next();
        auto *root_node_ptr = new Node(0, compute_internal_end_index(vp.n_branches), std::vector < int > {});
        auto *buffer_ptr = new Buffer(root_node_ptr);
        while(buffer_ptr->len() > 0) {
            auto *hypercubes = compute_vertex_velocities_of_sub_problem(vp.computation_parameters.number_of_cubes_per_gpu_call,
                                                                        buffer_ptr,
                                                                        lambda_ranges);

            const odesolver::DevDatC vertices = hypercubes->get_vertices();
            const odesolver::DevDatC vertex_velocities = hypercubes->get_vertex_velocities();

            if (!skip_fix_lambdas) {
                write_data_to_ofstream(vertex_velocities, os);
                if (with_vertices)
                    write_data_to_ofstream(vertices, os_vertices);
            } else {
                write_data_to_ofstream(vertex_velocities, os, skip_iterators_in_dimensions);
                if (with_vertices)
                    write_data_to_ofstream(vertices, os_vertices, skip_iterators_in_dimensions);
            }

            /* for(auto dim_index = 0; dim_index < dim; dim_index++)
            {
                // print_range("Vertices in dimension " + std::to_string(dim_index + 1), vertices[dim_index]->begin(), vertices[dim_index]->end());
                // print_range("Vertex velocities in dimension " + std::to_string(dim_index + 1), vertex_velocities[dim_index]->begin(), vertex_velocities[dim_index]->end());
            }*/

            delete hypercubes;
        }

        delete buffer_ptr;
        delete root_node_ptr;

        c++;
    }
    os.close();
    if(with_vertices)
        os_vertices.close();

    // Consistency check
    auto total_number_of_generated_lambdas = 1;
    for(auto &fix_lambd : vp.fix_lambdas)
        total_number_of_generated_lambdas *= fix_lambd.size();
    if(total_number_of_generated_lambdas != c)
    {
        std::cout << "\nERROR: Something went wrong during the generation of the lambda ranges (possibility for duplicates, etc.)" << vp.dim << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Visualization::compute_vertex_velocities_from_parameters(std::string dir)
{
    auto compute_vertex_velocities_parameters = vp.get_value_by_key<json>("compute_vertex_velocities");
    auto params1 = VisualizationParameters::ComputeVertexVelocitiesParameters(compute_vertex_velocities_parameters);
    compute_vertex_velocities(dir, params1.skip_fix_lambdas, params1.with_vertices);
}

void Visualization::compute_separatrizes(const std::string dir,
                          const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges,
                          const std::vector <cudaT> minimum_change_of_state,
                          const cudaT minimum_delta_t, const cudaT maximum_flow_val,
                          const std::vector <cudaT> vicinity_distances,
                          const uint observe_every_nth_step, const uint maximum_total_number_of_steps,
                          const uint N_per_eigen_dim,
                          const std::vector<double> shift_per_dim)
{
    cudaT delta_t = 0.001;

    // Get saddle fixed points (in coordinates)
    // Sample around fix points
    // Evolve with corresponding particularities

    // odesolver::DevDatC saddle_coordinates = get_saddle_fix_points();

    std::vector < std::vector<cudaT> > potential_saddle_points = vp.get_fixed_points();
    // std::vector < std::vector< std::vector <int> > > eigen_compositions = vp.get_eigen_compositions();

    CoordinateOperatorParameters coordinate_operator_parameters = CoordinateOperatorParameters::from_parameters(vp.path_parameters.theory, potential_saddle_points);
    CoordinateOperator saddle_point_evaluator(coordinate_operator_parameters);
    saddle_point_evaluator.compute_jacobians_and_eigendata();
    /* auto eigenvectors_real_part = saddle_point_evaluator.get_real_parts_of_eigenvectors();
    auto eigenvectors_imag_part = saddle_point_evaluator.get_imag_parts_of_eigenvectors();
    auto eigenvalues = saddle_point_evaluator.get_real_parts_of_eigenvalues(); */

    const std::vector<int> saddle_point_indices = saddle_point_evaluator.get_indices_with_saddle_point_characteristics();

    // Iterator through emerging lambda ranges from given fixed lambdas
    LambdaRangeGenerator lambda_range_generator(vp.n_branches, vp.partial_lambda_ranges, vp.fix_lambdas);
    auto c = 0;
    while(!lambda_range_generator.finished())
    {
        std::ofstream os;
        std::string path = vp.get_absolute_path(vp.path_parameters.get_base_path() + "/" + dir + "/", vp.path_parameters.relative_path);
        os.open(path + "separatrices_" + std::to_string(c) + ".dat");

        auto lambda_ranges = lambda_range_generator.next();

        // Retrieve currently considered fixed lambdas
        std::vector< cudaT > fixed_lambdas {};
        std::transform(indices_of_fixed_lambdas.begin(), indices_of_fixed_lambdas.end(), std::back_inserter(fixed_lambdas),
                       [lambda_ranges] (const int& index) { return lambda_ranges[index].first; });
        // Iterate over all saddle points
        for(auto saddle_point_index = 0; saddle_point_index < saddle_point_indices.size(); saddle_point_index++)
        {
            std::cout << "Performing for saddle point with x = " << potential_saddle_points[saddle_point_index][0] << std::endl;
            /* std::vector<std::vector<cudaT>> eigenvector_real_part = eigenvectors_real_part[saddle_point_index];
            std::vector<std::vector<cudaT>> eigenvector_imag_part = eigenvectors_imag_part[saddle_point_index];
            std::vector<cudaT> eigenvalue = eigenvalues[saddle_point_index]; */

            auto eigenvector = saddle_point_evaluator.get_eigenvector(saddle_point_index);
            auto eigenvalue = saddle_point_evaluator.get_eigenvalue(saddle_point_index);

            std::vector<int> stable_manifold_indices {};
            std::vector<int> unstable_manifold_indices {};
            std::vector<std::vector<cudaT>> manifold_eigenvectors(eigenvalue.size());
            extract_stable_and_unstable_manifolds(eigenvalue, eigenvector, stable_manifold_indices, unstable_manifold_indices, manifold_eigenvectors);

            compute_separatrizes_of_manifold(
                    potential_saddle_points[saddle_point_index],
                    stable_manifold_indices,
                    manifold_eigenvectors,
                    -1.0*delta_t,
                    boundary_lambda_ranges,
                    minimum_change_of_state,
                    minimum_delta_t, maximum_flow_val,
                    vicinity_distances,
                    observe_every_nth_step, maximum_total_number_of_steps,
                    N_per_eigen_dim,
                    shift_per_dim,
                    os,
                    fixed_lambdas
            );

            compute_separatrizes_of_manifold(
                    potential_saddle_points[saddle_point_index],
                    unstable_manifold_indices,
                    manifold_eigenvectors,
                    delta_t,
                    boundary_lambda_ranges,
                    minimum_change_of_state,
                    minimum_delta_t, maximum_flow_val,
                    vicinity_distances,
                    observe_every_nth_step, maximum_total_number_of_steps,
                    N_per_eigen_dim,
                    shift_per_dim,
                    os,
                    fixed_lambdas
            );
        }

        os.close();
        c++;
    }

    // Consistency check
    auto total_number_of_generated_lambdas = 1;
    for(auto &fix_lambd : vp.fix_lambdas)
        total_number_of_generated_lambdas *= fix_lambd.size();
    if(total_number_of_generated_lambdas != c) {
        std::cout
                << "\nERROR: Something went wrong during the generation of the lambda ranges (possibility for duplicates, etc.)"
                << vp.dim << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Visualization::compute_separatrizes_from_parameters(const std::string dir)
{
    // ToDo: Reactivate
    /* auto evolve_on_condition_parameters = vp.get_value_by_key<json>("evolve_on_condition");
    auto compute_separatrizes_parameters = vp.get_value_by_key<json>("compute_separatrizes");
    auto params3 = VisualizationParameters::ComputeSeparatrizesParameters(compute_separatrizes_parameters);
    if(vp.fix_lambdas.size() > 0)
    {
        auto conditional_observer_parameters = vp.get_value_by_key<json>("conditional_intersection_observer");
        auto params1 = ConditionalIntersectionObserverParameters(conditional_observer_parameters);
        auto params2 = CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalIntersectionObserverParameters>(evolve_on_condition_parameters);
        compute_separatrizes(dir, params1.boundary_lambda_ranges, params1.minimum_change_of_state,
                             params1.minimum_delta_t, params1.maximum_flow_val, params1.vicinity_distances,
                             params2.observe_every_nth_step, params2.maximum_total_number_of_steps,
                             params3.N_per_eigen_dim, params3.shift_per_dim);
    }
    else
    {
        auto conditional_observer_parameters = vp.get_value_by_key<json>("conditional_range_observer");
        auto params1 = ConditionalRangeObserverParameters(conditional_observer_parameters);
        auto params2 = CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalRangeObserverParameters>(evolve_on_condition_parameters);
        compute_separatrizes(dir, params1.boundary_lambda_ranges, params1.minimum_change_of_state,
                             params1.minimum_delta_t, params1.maximum_flow_val, std::vector <cudaT> {},
                             params2.observe_every_nth_step, params2.maximum_total_number_of_steps,
                             params3.N_per_eigen_dim, params3.shift_per_dim);
    } */
}


// Function is very similar to main function of fixed_point_search -> integrate them somehow??
HyperCubes * Visualization::compute_vertex_velocities_of_sub_problem(
        const int number_of_cubes_per_gpu_call,
        Buffer * buffer_ptr,
        const std::vector <std::pair<cudaT, cudaT> > lambda_ranges)
{
    std::vector< Node* > node_package;
    int total_number_of_cubes = 0;
    int maximum_depth = 0;

    // Get nodes for the gpu from buffer
    std::tie(node_package, total_number_of_cubes, maximum_depth) = buffer_ptr->get_first_nodes(number_of_cubes_per_gpu_call);

    if(monitor) {
        std::cout << "\n### Nodes for the qpu: " << node_package.size() << ", total number of cubes: "
                  << total_number_of_cubes << std::endl;
        buffer_ptr->get_nodes_info(node_package);
    }

    auto * hypercubes_ptr = new HyperCubes(vp.k, std::vector< std::vector<int> > {vp.n_branches}, lambda_ranges);

    // Use helper class to perform gpu tasks on nodes
    GridComputationWrapper grcompwrap = hypercubes_ptr->generate_and_linearize_nodes(total_number_of_cubes, maximum_depth, node_package);

    // Compute the actual vertices by first expanding each cube according to the number of vertices to
    // a vector of reference vertices of length total_number_of_cubes*dim and then computing the indices
    hypercubes_ptr->compute_reference_vertices(grcompwrap);

    // hypercubes.test_projection();
    // Compute vertex velocities
    hypercubes_ptr->determine_vertex_velocities(vp.flow_equations);

    return hypercubes_ptr;
}

odesolver::DevDatC Visualization::sample_around_saddle_point(const std::vector<double> coordinate, const std::vector<int> manifold_indices,
                                                  const std::vector<std::vector<cudaT>> manifold_eigenvectors, const std::vector<double> shift_per_dim, const uint N_per_eigen_dim)
{
    const uint eigen_dim = manifold_indices.size();
    const int N = pow(N_per_eigen_dim, eigen_dim);
    const uint8_t dim = coordinate.size();
    odesolver::DevDatC sampled_points(dim, N, 0);

    // Generate (eigen_dim x N) random numbers
    int discard = 0;
    odesolver::DevDatC random_numbers(eigen_dim, N, 0);
    for(auto eigen_dim_index = 0; eigen_dim_index < eigen_dim; eigen_dim_index++) {
        thrust::transform(
                thrust::make_counting_iterator(0 + discard),
                thrust::make_counting_iterator(N + discard),
                random_numbers[eigen_dim_index].begin(),
                RandomNormalGenerator());
        discard += N;
    }

    dev_vec sum(N, 0);
    for(auto dim_index=0; dim_index < dim; dim_index++)
    {
        // Iteration over the random_numbers per eigen_dimension
        for(auto eigen_dim_index = 0; eigen_dim_index < eigen_dim; eigen_dim_index++)
        {
            thrust::transform(sampled_points[dim_index].begin(), sampled_points[dim_index].end(), random_numbers[eigen_dim_index].begin(), sampled_points[dim_index].begin(),
                    sum_manifold_eigenvector(manifold_eigenvectors[manifold_indices[eigen_dim_index]][dim_index]));
        }

        // For latter normalization
        thrust::transform(sum.begin(), sum.end(), sampled_points[dim_index].begin(), sum.begin(), sum_square());
    }

    for(auto dim_index=0; dim_index < dim; dim_index++)
        thrust::transform(sampled_points[dim_index].begin(), sampled_points[dim_index].end(), sum.begin(), sampled_points[dim_index].begin(), normalize_by_square_root());
    // Shift coordinates by random numbers
    for(auto dim_index=0; dim_index < dim; dim_index++)
    {
        // if(std::find(manifold_indices.begin(), manifold_indices.end(), dim_index) != manifold_indices.end())
        // print_range("Sampled points ", sampled_points[eigen_dim_index].begin(), sampled_points[eigen_dim_index].end());
        thrust::transform(
                sampled_points[dim_index].begin(),
                sampled_points[dim_index].end(),
                sampled_points[dim_index].begin(),
                finalize_sample_around_saddle_point(coordinate[dim_index], shift_per_dim[dim_index]));
        // print_range("Sampled points2 ", sampled_points[eigen_dim_index].begin(), sampled_points[eigen_dim_index].end());
    }
    return sampled_points;
}

odesolver::DevDatC Visualization::get_initial_values_to_eigenvector(const std::vector<double> saddle_point, const std::vector<cudaT> eigenvector, const std::vector<double> shift_per_dim)
{
    const uint8_t dim = saddle_point.size();
    odesolver::DevDatC points(dim, 2, 0);

    for(auto dim_index=0; dim_index < dim; dim_index++)
    {
        auto it = points[dim_index].begin();
        *it = saddle_point[dim_index] + shift_per_dim[dim_index] * eigenvector[dim_index];
        it++;
        *it = saddle_point[dim_index] - shift_per_dim[dim_index] * eigenvector[dim_index];
        // print_range("Points ", points[dim_index].begin(), points[dim_index].end());
    }
    return points;
}


void Visualization::extract_stable_and_unstable_manifolds(
        Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType eigenvalue,
        Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType eigenvector,
        std::vector<int> &stable_manifold_indices,
        std::vector<int> &unstable_manifold_indices,
        std::vector<std::vector<cudaT>> &manifold_eigenvectors
)
{
    std::vector< std::complex<double> > complex_eigenvals_buffer{};
    std::vector< int > complex_eigenvals_buffer_index{};

    for(auto i = 0; i < eigenvalue.size(); i++)
    {
        if(eigenvalue[i].real() < 0)
            stable_manifold_indices.push_back(i);
        else
            unstable_manifold_indices.push_back(i);

        auto eigen_vec = eigenvector.col(i);
        if(eigenvalue[i].imag() != 0)
        {
            auto it_complex_eigenvals_buffer = std::find(complex_eigenvals_buffer.begin(), complex_eigenvals_buffer.end(), std::conj(eigenvalue[i]));
            if (it_complex_eigenvals_buffer != complex_eigenvals_buffer.end())
            {
                for(auto j = 0; j < eigen_vec.size(); j++)
                {
                    manifold_eigenvectors[complex_eigenvals_buffer_index[it_complex_eigenvals_buffer - complex_eigenvals_buffer.begin()]].push_back(eigen_vec[j].imag());
                    manifold_eigenvectors[i].push_back(eigen_vec[j].real());
                }
            }
            else
            {
                complex_eigenvals_buffer.push_back(eigenvalue[i]);
                complex_eigenvals_buffer_index.push_back(i);
            }
        }
        else
        {
            for(auto j = 0; j < eigen_vec.size(); j++)
                manifold_eigenvectors[i].push_back(eigen_vec[j].real());
        }
    }
}


void Visualization::compute_separatrizes_of_manifold(
        const std::vector<double> saddle_point,
        const std::vector<int> manifold_indices,
        const std::vector<std::vector<cudaT>> manifold_eigenvectors,
        const cudaT delta_t,
        const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges,
        const std::vector <cudaT> minimum_change_of_state,
        const cudaT minimum_delta_t, const cudaT maximum_flow_val,
        const std::vector <cudaT> vicinity_distances,
        const uint observe_every_nth_step, const uint maximum_total_number_of_steps,
        const uint N_per_eigen_dim,
        const std::vector<double> shift_per_dim,
        std::ofstream &os,
        std::vector< cudaT > fixed_lambdas
        ) {
    // ToDo: Reactivate
    // Perform computation of separatrix for stable manifold
    /* odesolver::DevDatC sampled_coordinates;
    // Single line
    if(manifold_indices.size() == 1)
    {
        sampled_coordinates = get_initial_values_to_eigenvector(saddle_point,
                                                                manifold_eigenvectors[manifold_indices[0]],
                                                                shift_per_dim);
    }
    else
    {
        sampled_coordinates = sample_around_saddle_point(saddle_point,
                                                         manifold_indices,
                                                         manifold_eigenvectors,
                                                         shift_per_dim, N_per_eigen_dim);
    }

    if(vp.fix_lambdas.size() > 0)
    {
        auto observer = new ConditionalIntersectionObserver(vp.flow_equations, sampled_coordinates.size(), os,
                                                            boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val, vicinity_distances,
                                                            fixed_lambdas, indices_of_fixed_lambdas);
        observer->initalize_side_counter(sampled_coordinates);

        // for(auto dim_index = 0; dim_index < sampled_coordinates.dim_size(); dim_index++)
        //     print_range("Vertex in dim " + std::to_string(dim_index) + ": ", sampled_coordinates[dim_index].begin(), sampled_coordinates[dim_index].end());
        Evolution<ConditionalIntersectionObserver> evaluator(vp.flow_equations, observer, observe_every_nth_step, maximum_total_number_of_steps);

        print_range("Initial point", sampled_coordinates.begin(), sampled_coordinates.end());
        evaluator.evolve_observer_based(sampled_coordinates, delta_t);
        print_range("End point", sampled_coordinates.begin(), sampled_coordinates.end());
    }
    else
    {
        auto observer = new ConditionalRangeObserver(vp.flow_equations, sampled_coordinates.size(), os,
                                                     boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val);
        // for(auto dim_index = 0; dim_index < sampled_coordinates.dim_size(); dim_index++)
        //     print_range("Vertex in dim " + std::to_string(dim_index) + ": ", sampled_coordinates[dim_index].begin(), sampled_coordinates[dim_index].end());
        Evolution<ConditionalRangeObserver> evaluator(vp.flow_equations, observer, observe_every_nth_step, maximum_total_number_of_steps);

        print_range("Initial point", sampled_coordinates.begin(), sampled_coordinates.end());
        evaluator.evolve_observer_based(sampled_coordinates, delta_t);
        print_range("End point", sampled_coordinates.begin(), sampled_coordinates.end());
    } */
}