#ifndef LORENTZ_ATTRACTOR_FIXED_POINT_SEARCH_T_HPP
#define LORENTZ_ATTRACTOR_FIXED_POINT_SEARCH_T_HPP

#include <chrono>

#include <odesolver/modes/fixed_point_search.hpp>
#include <odesolver/modes/coordinate_operator.hpp>

#include "../flow_equations/lorentz_attractor/lorentz_attractor_flow_equation.hpp"
#include "../flow_equations/lorentz_attractor/lorentz_attractor_jacobian_equation.hpp"

void find_fixed_points();

void evaluate_fixed_points();

#endif //LORENTZ_ATTRACTOR_FIXED_POINT_SEARCH_T_HPP
