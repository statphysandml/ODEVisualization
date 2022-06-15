#ifndef THREE_POINT_SYSTEM_FIXED_POINT_SEARCH_T_HPP
#define THREE_POINT_SYSTEM_FIXED_POINT_SEARCH_T_HPP

#include <chrono>

#include <odesolver/modes/recursive_search.hpp>
#include <odesolver/recursive_search/fixed_point_criterion.hpp>
#include <odesolver/modes/kmeans_clustering.hpp>


#include "../flow_equations/three_point_system/three_point_system_flow_equation.hpp"
#include "../flow_equations/three_point_system/three_point_system_jacobian_equation.hpp"

// ToDo: Adapt this to the current code
/* FixedPointSearch build_fixed_point_search_parameters();

FixedPointSearch::ClusterParameters build_cluster_parameters();

// Add parameters for the clustering of solutions to the config file
void add_cluster_parameters_to_fixed_point_search_to_file();

void write_fixed_point_search_params_to_file(const std::string dir = "example_fixed_point_search");

void fixed_point_search();

void fixed_points_search(); */

void find_fixed_points();

#endif //THREE_POINT_SYSTEM_FIXED_POINT_SEARCH_T_HPP
