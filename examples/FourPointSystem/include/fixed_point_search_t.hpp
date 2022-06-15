#ifndef FOUR_POINT_SYSTEM_FIXED_POINT_SEARCH_T_HPP
#define FOUR_POINT_SYSTEM_FIXED_POINT_SEARCH_T_HPP

#include <chrono>

#include <odesolver/modes/recursive_search.hpp>
#include <odesolver/recursive_search/fixed_point_criterion.hpp>
#include <odesolver/modes/kmeans_clustering.hpp>


#include "../flow_equations/four_point_system/four_point_system_flow_equation.hpp"
#include "../flow_equations/four_point_system/four_point_system_jacobian_equation.hpp"

void find_fixed_points();

#endif //FOUR_POINT_SYSTEM_FIXED_POINT_SEARCH_T_HPP
