#ifndef LORENTZ_ATTRACTOR_FLOW_OBSERVER_T_HPP
#define LORENTZ_ATTRACTOR_FLOW_OBSERVER_T_HPP

#include <cmath>

#include <thrust/transform.h>

#include <odesolver/dev_dat.hpp>
#include <odesolver/evolution/evolution_observer.hpp>
#include <odesolver/util/thrust_functors.hpp>

#include "../flow_equations/lorentz_attractor/lorentz_attractor_flow_equation.hpp"
#include "../flow_equations/lorentz_attractor/lorentz_attractor_jacobian_equation.hpp"

#include "dev_dat_t.hpp"

void no_change_t();

void divergent_flow_t();

void out_of_range_t();

void out_of_range_t2();

void intersection_observer_t();

void trajectory_observer_t();

void evolution_observer_t();

#endif //LORENTZ_ATTRACTOR_FLOW_OBSERVER_T_HPP