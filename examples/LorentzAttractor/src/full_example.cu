#include "../include/full_example.hpp"


void find_fixed_points()
{
    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {
        std::vector<int> {100, 100, 100}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
        std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
        std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
        std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
        std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
        std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}
    };
    // mu, lambda, g
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
        std::pair<cudaT, cudaT> (-12.0, 12.0), std::pair<cudaT, cudaT> (-12.0, 12.0), std::pair<cudaT, cudaT> (-1.0, 31.0)};

    auto fixed_point_search = FixedPointSearch::from_parameters(
        maximum_recursion_depth,
        n_branches_per_depth,
        lambda_ranges,
        generate_flow_equations<LorentzAttractorFlowEquations>(0)
    );

    // Setting gpu specfic computation parameters (optional) - parameters are already set default
    const int number_of_cubes_per_gpu_call = 100;
    const int maximum_number_of_gpu_calls = 100000;
    fixed_point_search.set_computation_parameters(
        number_of_cubes_per_gpu_call,
        maximum_number_of_gpu_calls
    );

    // Find fixed point solutions
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();

    fixed_point_search.find_fixed_points_dynamic_memory();
    // fixed_point_search.find_fixed_points_preallocated_memory();

    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);
    std::cout << fs.count() << "s\n";
    std::cout << d.count() << "ms\n";

    NodeCounter<Node>::print_statistics();

    // Just for testing issues -> get solutions and print infos about these
    std::vector<std::shared_ptr<Leaf>> solutions = fixed_point_search.get_solutions();
    for(auto &sol: solutions)
        sol->info();

    // Explicit use of parameters for clustering
    // const std::string dir = "fixed_point_search_interface";
    const uint maximum_expected_number_of_clusters = 80;
    const double upper_bound_for_min_distance = 0.01;
    const uint maximum_number_of_iterations = 1000;

    // Cluster solutions
    fixed_point_search.cluster_solutions_to_fixed_points(
        maximum_expected_number_of_clusters,
        upper_bound_for_min_distance,
        maximum_number_of_iterations
    );
    
    fixed_point_search.write_fixed_points_to_file("data/fe_fixed_point_search");
}

void evaluate_fixed_points()
{
    auto fixed_points = load_fixed_points("data/fe_fixed_point_search");

    CoordinateOperator evaluator = CoordinateOperator::from_vecvec(
        fixed_points,
        generate_flow_equations<LorentzAttractorFlowEquations>(0),
        generate_jacobian_equations<LorentzAttractorJacobianEquations>(0)
    );

    evaluator.compute_velocities();
    evaluator.compute_jacobians();

    evaluator.write_characteristics_to_file("data/fe_fixed_point_characteristics");
}