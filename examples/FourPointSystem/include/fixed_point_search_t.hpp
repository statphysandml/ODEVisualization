#include <odesolver/fixed_point_search.hpp>
#include <odesolver/coordinate_operator.hpp>

#include "../flow_equations/four_point_system/four_point_system_flow_equation.hpp"
#include "../flow_equations/four_point_system/four_point_system_jacobian_equation.hpp"

FixedPointSearch build_fixed_point_search_parameters();

FixedPointSearch::ClusterParameters build_cluster_parameters();

// Add parameters for the clustering of solutions to the config file
void add_cluster_parameters_to_fixed_point_search_to_file();

void write_fixed_point_search_params_to_file(const std::string dir = "example_fixed_point_search");

void fixed_point_search();

void fixed_points_search();

void find_fixed_points();

void evaluate_fixed_points();