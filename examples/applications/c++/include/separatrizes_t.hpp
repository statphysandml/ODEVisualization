#ifndef LORENTZ_ATTRACTOR_SEPARATRIZES_T_HPP
#define LORENTZ_ATTRACTOR_SEPARATRIZES_T_HPP

#include <cmath>

#include <thrust/transform.h>

#include <devdat/devdat.hpp>
#include <devdat/util/thrust_functors.hpp>

#include <odesolver/evolution/evolution_observer.hpp>
#include <odesolver/evolution/stepper.hpp>
#include <odesolver/modes/separatrizes.hpp>

#include <lorentz_attractor/lorentz_attractor_flow_equation.hpp>
#include <lorentz_attractor/lorentz_attractor_jacobian_equation.hpp>

#include "dev_dat_t.hpp"

void separatrizes_t();


#endif //LORENTZ_ATTRACTOR_SEPARATRIZES_T_HPP