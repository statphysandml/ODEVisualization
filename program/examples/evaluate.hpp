//
// Created by lukas on 31.03.20.
//

#ifndef PROGRAM_EVALUATE_HPP
#define PROGRAM_EVALUATE_HPP

#include "../ode_solver/include/coordinate_operator.hpp"

CoordinateOperatorParameters build_coordinate_operator_parameters(const std::string mode)
{
    const std::string theory ="three_point_system";

    std::vector < std::vector<cudaT> > coordinates {
            std::vector<cudaT> {-0.164467294607589, -0.164092870399133, 0.570242801808599},
            std::vector<cudaT> {-0.2, -0.2, 0.6}
    };

    CoordinateOperatorParameters evaluator_parameters = CoordinateOperatorParameters::from_parameters(theory, coordinates, mode);
    return evaluator_parameters;
}

void write_coordinate_operator_params_to_file(const std::string dir, const std::string mode)
{
    CoordinateOperatorParameters evaluator_parameters = build_coordinate_operator_parameters(mode);
    evaluator_parameters.write_to_file(dir);
}

// Mode: "jacobian"
void run_evaluate_velocities_and_jacobians_from_file()
{
    const std::string theory ="three_point_system";
    const std::string mode = "jacobian";
    const std::string dir = "example_evaluate_velocities_and_jacobians";

    // Load existing parameter file
    CoordinateOperatorParameters evaluator_parameters = CoordinateOperatorParameters::from_file(theory, mode, dir);

    // (In principle four options)
    Executer::exec_evaluate(evaluator_parameters, dir);
}

void evaluate_velocities_and_jacobians()
{
    // Write necessary parameter files to file
    write_coordinate_operator_params_to_file("example_evaluate_velocities_and_jacobians", "jacobian");

    // Run
    run_evaluate_velocities_and_jacobians_from_file();
}

void evolve_a_hypercube()
{
    // ToDo! Evolve a set of hypercubes
}


#endif //PROGRAM_EVALUATE_HPP
