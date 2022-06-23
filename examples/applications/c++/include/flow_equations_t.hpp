#ifndef LORENTZ_ATTRACTOR_FLOW_EQUATIONS_T_HPP
#define LORENTZ_ATTRACTOR_FLOW_EQUATIONS_T_HPP

#include <cmath>

#include <thrust/transform.h>

#include <devdat/devdat.hpp>

#include <lorentz_attractor/lorentz_attractor_flow_equation.hpp>
#include <lorentz_attractor/lorentz_attractor_jacobian_equation.hpp>

devdat::DevDatC get_fixed_points();

void compute_flow_t();

devdat::DevDatC compute_jacobians();

std::vector<std::vector<double>> compute_jacobian_elements_t();

#endif //LORENTZ_ATTRACTOR_FLOW_EQUATIONS_T_HPP