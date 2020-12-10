//
// Created by lukas on 31.03.20.
//

#ifndef PROGRAM_FIXED_POINT_SEARCH_WITHOUT_PARAM_FILES_HPP
#define PROGRAM_FIXED_POINT_SEARCH_WITHOUT_PARAM_FILES_HPP


#include "../ode_solver/include/fixed_point_search.hpp"
#include "fixed_point_search.hpp"


//[ Fixed point search //]

// Run fixed point search without storing parameters explicitly
void fixed_points_search()
{
    // See also execeuter - mode: fixed_point_search
    FixedPointSearchParameters fixed_point_search_parameters = build_fixed_point_search_parameters();

    FixedPointSearch fixed_point_search(fixed_point_search_parameters);
    // Find fixed point solutions
    fixed_point_search.find_fixed_point_solutions();
    NodeCounter<Node>::print_statistics();

    // Just for testing issues -> get solutions and print infos about these
    std::vector<Leaf *> solutions = fixed_point_search.get_solutions();
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
}

#include "../examples/evaluate.hpp"

#include "../ode_solver/include/coordinate_operator.hpp"

void evaluate_velocities_and_jacobians_of_coordinates()
{
    CoordinateOperatorParameters evaluator_parameters = build_coordinate_operator_parameters("jacobian");

    CoordinateOperator evaluator(evaluator_parameters);
    evaluator.compute_velocities();
    evaluator.compute_jacobians_and_eigendata();
}

#endif //PROGRAM_FIXED_POINT_SEARCH_WITHOUT_PARAM_FILES_HPP