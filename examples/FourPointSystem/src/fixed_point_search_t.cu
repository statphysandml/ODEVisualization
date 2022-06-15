#include "../include/fixed_point_search_t.hpp"


void find_fixed_points()
{
    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {
            std::vector<int> {10, 10, 10, 10, 10}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2},
            std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2},
            std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2},
            std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2},
            std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2},
            std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}};
    // mu, Lam3, Lam4, g3, g4
    const std::vector <std::pair<cudaT, cudaT> > variable_ranges = std::vector <std::pair<cudaT, cudaT> > {
        std::pair<cudaT, cudaT> (-2.0, 2.0),
        std::pair<cudaT, cudaT> (-1.0, 1.0),
        std::pair<cudaT, cudaT> (-2.0, 2.0),
        std::pair<cudaT, cudaT> (0.05, 2.0),
        std::pair<cudaT, cudaT> (0.05, 2.0)
    };

    std::shared_ptr<odesolver::recursivesearch::RecursiveSearchCriterion> recursive_search_criterion_ptr = std::make_unique<odesolver::recursivesearch::FixedPointCriterion>();

    auto fixed_point_search = odesolver::modes::RecursiveSearch::generate(
        maximum_recursion_depth,
        n_branches_per_depth,
        variable_ranges,
        recursive_search_criterion_ptr,
        odesolver::flowequations::generate_flow_equations<FourPointSystemFlowEquations>(0),
        nullptr,
        400000,
        1000
    );

    // Find fixed point solutions
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();

    fixed_point_search.eval();
    // fixed_point_search.eval("preallocated_memory");
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

    // Cluster solutions
    auto kmeans_clustering = odesolver::modes::KMeansClustering::generate(
        10, // maximum_expected_number_of_clusters
        0.01, // upper_bound_for_min_distance
        1000 // maximum_number_of_iterations
    );
    auto fixed_points = kmeans_clustering.eval(fixed_point_search.solutions());
    
    // fixed_points.write_to_file("data/fe_fixed_point_search", "fixed_points");
}