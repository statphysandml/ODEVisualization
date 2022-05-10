#ifndef LORENTZ_ATTRACTOR_FLOW_EQUATIONS_T_HPP
#define LORENTZ_ATTRACTOR_FLOW_EQUATIONS_T_HPP

#include <cmath>

#include <thrust/transform.h>

#include <odesolver/dev_dat.hpp>

#include "../flow_equations/lorentz_attractor/lorentz_attractor_flow_equation.hpp"
#include "../flow_equations/lorentz_attractor/lorentz_attractor_jacobian_equation.hpp"

odesolver::DevDatC get_fixed_points();

void compute_vertex_velocities_t();

odesolver::DevDatC compute_jacobians();

std::vector<std::vector<double>> compute_jacobian_elements_t();

#endif //LORENTZ_ATTRACTOR_FLOW_EQUATIONS_T_HPP