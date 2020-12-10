//
// Created by lukas on 01.04.20.
//

#ifndef PROGRAM_EVOLVE_HPP
#define PROGRAM_EVOLVE_HPP

#include "evaluate.hpp"
#include "../ode_solver/include/coordinate_operator.hpp"
#include "../ode_solver/include/observers/evolution.hpp"

EvolutionParameters build_evolve_parameters()
{
    const cudaT start_t=0.0;
    const cudaT end_t=1.0;
    const cudaT delta_t=0.01;
    std::string step_size_type="constant";
    const std::string results_dir="";

    EvolutionParameters evolve_parameters(start_t, end_t, delta_t, step_size_type, results_dir);

    return evolve_parameters;
}

// Add parameters to config file
void add_evolve_parameters_to_coordinate_operator_to_file(const std::string dir)
{
    const std::string theory = "three_point_system";
    const std::string mode = "evolve";

    // Load existing fixed_point_search parameter file
    CoordinateOperatorParameters coordinate_operator_parameters = CoordinateOperatorParameters::from_file(theory, mode, dir);

    // Parameters for clustering the resulting solutions - Represent parameters of a function
    EvolutionParameters evolve_parameters = build_evolve_parameters();
    coordinate_operator_parameters.set_evolution_params(evolve_parameters);
    coordinate_operator_parameters.write_to_file(dir);
}

void run_evolve_from_file()
{
    const std::string theory = "three_point_system";
    const std::string mode = "evolve";
    const std::string dir = "example_evolve";

    // Load existing fixed_point_search parameter file
    CoordinateOperatorParameters coordinate_operator_parameters = CoordinateOperatorParameters::from_file(theory, mode, dir);

    PathParameters path_parameters = coordinate_operator_parameters.get_path_parameters();
    Executer executer(path_parameters);
    executer.main(dir);
}

// Mode: "fixed_point_search"
void evolve()
{
    // Write necessary parameter files to file
    write_coordinate_operator_params_to_file("example_evolve", "evolve");
    add_evolve_parameters_to_coordinate_operator_to_file("example_evolve");

    // Run
    run_evolve_from_file();
}

#endif //PROGRAM_EVOLVE_HPP
