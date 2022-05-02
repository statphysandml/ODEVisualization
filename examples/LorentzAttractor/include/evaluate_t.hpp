#ifndef LORENTZ_ATTRACTOR_EVALUATE_T_HPP
#define LORENTZ_ATTRACTOR_EVALUATE_T_HPP

#include <odesolver/coordinate_operator.hpp>

#include "../flow_equations/lorentz_attractor/lorentz_attractor_flow_equation.hpp"
#include "../flow_equations/lorentz_attractor/lorentz_attractor_jacobian_equation.hpp"

CoordinateOperator build_coordinate_operator_parameters();

void write_coordinate_operator_params_to_file(const std::string rel_dir);

// Mode: "jacobian"
void run_evaluate_velocities_and_jacobians_from_file();

void evaluate_velocities_and_jacobians();

/* void evolve_a_hypercube()
{
    // ToDo! Evolve a set of hypercubes
} */

#endif //LORENTZ_ATTRACTOR_EVALUATE_T_HPP
