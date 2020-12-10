//
// Created by lukas on 31.03.20.
//

#ifndef PROGRAM_FIXED_POINT_SEARCH_HPP
#define PROGRAM_FIXED_POINT_SEARCH_HPP


#include "../ode_solver/include/executer.hpp"


FixedPointSearchParameters build_fixed_point_search_parameters()
{
    const std::string theory = "three_point_system";

    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {
            std::vector<int> {20, 20, 20}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2},
            std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};
    // mu, lambda, g
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};

    FixedPointSearchParameters fixed_point_search_parameters(
            theory,
            maximum_recursion_depth,
            n_branches_per_depth,
            lambda_ranges);

    // Setting gpu specfic computation parameters (optional) - parameters are already set default
    const int number_of_cubes_per_gpu_call = 20000;
    const int maximum_number_of_gpu_calls = 1000;
    fixed_point_search_parameters.set_computation_parameters(
            number_of_cubes_per_gpu_call,
            maximum_number_of_gpu_calls);

    return fixed_point_search_parameters;
}

FixedPointSearchParameters::ClusterParameters build_cluster_parameters()
{
    const uint maximum_expected_number_of_clusters = 80;
    const double upper_bound_for_min_distance = 0.01;
    const uint maximum_number_of_iterations = 1000;
    FixedPointSearchParameters::ClusterParameters cluster_parameters(maximum_expected_number_of_clusters,
                                                                     upper_bound_for_min_distance, maximum_number_of_iterations);
    return cluster_parameters;
}

// Add parameters for the clustering of solutions to the config file
void add_cluster_parameters_to_fixed_point_search_to_file()
{
    const std::string theory = "three_point_system";
    const std::string dir = "example_fixed_point_search";
    const std::string mode = "fixed_point_search";
    // Load existing fixed_point_search parameter file
    FixedPointSearchParameters fixed_point_search_parameters(theory, mode, dir);

    // Parameters for clustering the resulting solutions - Represent parameters of a function
    FixedPointSearchParameters::ClusterParameters cluster_parameters = build_cluster_parameters();
    fixed_point_search_parameters.append_parameters(cluster_parameters);
    fixed_point_search_parameters.write_to_file(dir);
}

void write_fixed_point_search_params_to_file(const std::string dir = "example_fixed_point_search")
{
    FixedPointSearchParameters fixed_point_search_parameters = build_fixed_point_search_parameters();
    fixed_point_search_parameters.write_to_file(dir);
}

// Mode: "fixed_point_search"
void run_fixed_point_search_from_file()
{
    const std::string theory = "three_point_system";
    const std::string dir = "example_fixed_point_search";
    const std::string mode = "fixed_point_search";

    // Load existing fixed_point_search parameter file
    FixedPointSearchParameters fixed_point_search_parameters(theory, mode, dir);

    // Four options:


    // 1) Based on executer function
    // Executer::exec_fixed_point_search(fixed_point_search_parameters, dir);

    // 2) Another option to use executer modes (parameter file will be reloaded)
    PathParameters path_parameters = fixed_point_search_parameters.get_path_parameters();
    Executer executer(path_parameters);
    executer.main(dir);

    // 3) Based on custom code (similar to executer function)

    // 4) From console with: ./FRGVisualisation root_dir theory dir (does the same as 1)
}

void fixed_point_search()
{
    // Write necessary parameter files to file
    write_fixed_point_search_params_to_file("example_fixed_point_search");
    add_cluster_parameters_to_fixed_point_search_to_file();

    // Run
    run_fixed_point_search_from_file();
}


#endif //PROGRAM_FIXED_POINT_SEARCH_HPP
