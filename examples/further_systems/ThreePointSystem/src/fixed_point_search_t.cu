#include "../include/fixed_point_search_t.hpp"

// ToDo: Adapt this to the current code
/* FixedPointSearch build_fixed_point_search_parameters()
{
    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {
            std::vector<int> {20, 20, 20}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};
    // mu, lambda, g
    const std::vector <std::pair<cudaT, cudaT> > variable_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};

    auto fixed_point_search = FixedPointSearch::generate(
            maximum_recursion_depth,
            n_branches_per_depth,
            variable_ranges,
            flowequations::generate_flow_equations<ThreePointSystemFlowEquations>(0)
    );

    // Setting gpu specfic computation parameters (optional) - parameters are already set default
    fixed_point_search.set_computation_parameters(
        20000, // number_of_cubes_per_gpu_call
        1000 // maximum_number_of_gpu_calls
    );

    return fixed_point_search;
}

FixedPointSearch::ClusterParameters build_cluster_parameters()
{
    FixedPointSearch::ClusterParameters cluster_parameters(
        80, // maximum_expected_number_of_clusters
        0.01, // upper_bound_for_min_distance
        1000 // maximum_number_of_iterations
    );
    return cluster_parameters;
}

// Add parameters for the clustering of solutions to the config file
void add_cluster_parameters_to_fixed_point_search_to_file()
{
    const std::string rel_dir = "data/example_fixed_point_search/";

    // Load existing fixed_point_search parameter file
    auto fixed_point_search = FixedPointSearch::from_file(rel_dir, flowequations::generate_flow_equations<ThreePointSystemFlowEquations>(0));

    // Parameters for clustering the resulting solutions - Represent parameters of a function
    FixedPointSearch::ClusterParameters cluster_parameters = build_cluster_parameters();
    fixed_point_search.append_parameters(cluster_parameters);

    fixed_point_search.write_configs_to_file(rel_dir);
}

void write_fixed_point_search_params_to_file(const std::string rel_dir)
{
    FixedPointSearch fixed_point_search = build_fixed_point_search_parameters();
    fixed_point_search.write_configs_to_file(rel_dir);
}

// Mode: "fixed_point_search"
void run_fixed_point_search_from_file()
{
    const std::string rel_dir = "data/example_fixed_point_search/";
    

    // Load existing fixed_point_search parameter file
    auto fixed_point_search = FixedPointSearch::from_file(rel_dir, flowequations::<ThreePointSystemFlowEquations>(0));

    fixed_point_search.find_fixed_point_solutions();
    const std::string mode = "fixed_point_search"; // -> ToDo: Allow for adding a mode to the config file <-> allows for the computation with executer
    fixed_point_search.write_solutions_to_file(rel_dir);

    NodeCounter<Node>::print_statistics();

    // Cluster solutions
    fixed_point_search.cluster_solutions_to_fixed_points_from_file();
    fixed_point_search.write_fixed_points_to_file(rel_dir);
    // fixed_point_search.compute_and_write_fixed_point_characteristics_to_file(dir);

    // Four options:


    // 1) Based on executer function
    // Executer::exec_fixed_point_search(fixed_point_search_parameters, dir);

    // 2) Another option to use executer modes (parameter file will be reloaded)
    *//* PathParameters path_parameters = fixed_point_search_parameters.get_path_parameters();
    Executer executer(path_parameters);
    executer.main(dir); *//*

    // 3) Based on custom code (similar to executer function)

    // 4) From console with: ./FRGVisualisation root_dir theory dir (does the same as 1)
}

void fixed_point_search()
{
    // Write necessary parameter files to file
    write_fixed_point_search_params_to_file("data/example_fixed_point_search/");
    add_cluster_parameters_to_fixed_point_search_to_file();

    // Run
    run_fixed_point_search_from_file();
}

// Run fixed point search without storing parameters explicitly
void fixed_points_search()
{
    // See also execeuter - mode: fixed_point_search
    FixedPointSearch fixed_point_search = build_fixed_point_search_parameters();

    // Find fixed point solutions
    fixed_point_search.find_fixed_point_solutions();
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
            maximum_number_of_iterations);

    fixed_point_search.write_fixed_points_to_file("data/fixed_point_search");
} */

void find_fixed_points()
{
    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {
            std::vector<int> {20, 20, 20}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};
    // mu, lambda, g
    const std::vector <std::pair<cudaT, cudaT> > variable_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};

    std::shared_ptr<odesolver::recursivesearch::RecursiveSearchCriterion> recursive_search_criterion_ptr = std::make_unique<odesolver::recursivesearch::FixedPointCriterion>();

    auto fixed_point_search = odesolver::modes::RecursiveSearch::generate(
        maximum_recursion_depth,
        n_branches_per_depth,
        variable_ranges,
        recursive_search_criterion_ptr,
        flowequations::generate_flow_equations<ThreePointSystemFlowEquations>(0),
        nullptr,
        100,
        100000
    );

    // Find fixed point solutions
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();

    // fixed_point_search.eval("dynamic");
    fixed_point_search.eval("preallocated_memory");

    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);
    std::cout << fs.count() << "s\n";
    std::cout << d.count() << "ms\n";

    odesolver::collections::Counter<odesolver::collections::Collection>::print_statistics();

    // Just for testing issues -> get leaves and print infos about these
    std::vector<std::shared_ptr<odesolver::collections::Leaf>> leaves = fixed_point_search.leaves();
    for(auto &leaf: leaves)
        leaf->info();
    std::cout << "Found leaves " << fixed_point_search.leaves().size() << std::endl;

    // Cluster solutions
    auto kmeans_clustering = odesolver::modes::KMeansClustering::generate(
        10, // maximum_expected_number_of_clusters
        0.01, // upper_bound_for_min_distance
        1000 // maximum_number_of_iterations
    );
    auto fixed_points = kmeans_clustering.eval(fixed_point_search.solutions());
    
    // fixed_points.write_to_file("data/fe_fixed_point_search", "fixed_points");
}
