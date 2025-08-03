#ifndef LORENTZ_ATTRACTOR_FIXED_POINT_SEARCH_T_HPP
#define LORENTZ_ATTRACTOR_FIXED_POINT_SEARCH_T_HPP

#include <chrono>

#include <odesolver/modes/recursive_search.hpp>
#include <odesolver/recursive_search/fixed_point_criterion.hpp>
#include <odesolver/modes/kmeans_clustering.hpp>


#include <lorentz_attractor/lorentz_attractor_flow_equation.hpp>
#include <lorentz_attractor/lorentz_attractor_jacobian_equation.hpp>

void find_fixed_points();

#endif //LORENTZ_ATTRACTOR_FIXED_POINT_SEARCH_T_HPP
