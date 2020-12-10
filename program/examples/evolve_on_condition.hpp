//
// Created by lukas on 01.04.20.
//

#ifndef PROGRAM_EVOLVE_CONDITIONALLY_HPP
#define PROGRAM_EVOLVE_CONDITIONALLY_HPP

#include "../ode_solver/include/observers/conditional_range_observer.hpp"
#include "../ode_solver/include/observers/conditional_intersection_observer.hpp"

/* template<typename ConditionalObserverParameters>
CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalObserverParameters> build_evolve_on_condition_parameters()
{
    const uint observe_every_nth_step=10;
    const uint maximum_total_number_of_steps=1000000;
    const std::string results_dir="example_evolve_conditional_range";

    CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalObserverParameters> evolve_parameters(observe_every_nth_step, maximum_total_number_of_steps, results_dir);

    return evolve_parameters;
}

// Add parameters to config file
template<typename ConditionalObserverParameters>
void add_evolve_on_condition_parameters_to_coordinate_operator_to_file(const std::string dir)
{
    const std::string theory = "three_point_system";

    // Load existing fixed_point_search parameter file
    CoordinateOperatorParameters coordinate_operator_parameters(theory, dir);

    // Parameters for clustering the resulting solutions - Represent parameters of a function
    CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalObserverParameters> evolve_parameters = build_evolve_on_condition_parameters<ConditionalObserverParameters>();
    coordinate_operator_parameters.append_parameters(evolve_parameters);
    coordinate_operator_parameters.write_to_file(dir);
} */

/* ### Usage of conditional range ### */ 

/* ConditionalRangeObserverParameters build_conditional_range_observer_parameters()
{
    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-12, 1e-12, 1e-12};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    ConditionalRangeObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val);

    return conditional_observer_parameters;
}

// Add parameters to config file
void add_conditional_range_observer_parameters_to_coordinate_operator_to_file() {
    const std::string theory = "three_point_system";
    const std::string dir = "example_evolve_conditional_range";
    // Load existing fixed_point_search parameter file
    CoordinateOperatorParameters coordinate_operator_parameters(theory, dir);

    ConditionalRangeObserverParameters conditional_observer_parameters = build_conditional_range_observer_parameters();

    coordinate_operator_parameters.append_parameters(conditional_observer_parameters);
    coordinate_operator_parameters.write_to_file(dir);
}

void run_evolve_conditional_range_from_file()
{
    const std::string theory = "three_point_system";
    const std::string dir = "example_evolve_conditional_range";

    // Load existing fixed_point_search parameter file
    CoordinateOperatorParameters coordinate_operator_parameters(theory, dir);

    PathParameters path_parameters = coordinate_operator_parameters.get_path_parameters();
    Executer executer(path_parameters);
    executer.main(dir);
}

// Mode: "evolve_on_condition"
void evolve_conditional_range()
{
    // Write necessary parameter files to file
    write_coordinate_operator_params_to_file("example_evolve_conditional_range", "evolve_with_conditional_range_observer");
    add_conditional_range_observer_parameters_to_coordinate_operator_to_file();
    add_evolve_on_condition_parameters_to_coordinate_operator_to_file<ConditionalRangeObserverParameters>("example_evolve_conditional_range");

    // Run
    run_evolve_conditional_range_from_file();
}

/* ### Usage of conditional intersection ### */

/* ConditionalIntersectionObserverParameters build_conditional_intersection_observer_parameters()
{
    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-12, 1e-12, 1e-12};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    const std::vector <cudaT > vicinity_distances = std::vector< cudaT > {1e-5};
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val, vicinity_distances);

    return conditional_observer_parameters;
}

// Add parameters to config file
void add_conditional_intersection_observer_parameters_to_coordinate_operator_to_file()
{
    const std::string theory = "three_point_system";
    const std::string dir = "example_evolve_conditional_intersection";
    // Load existing fixed_point_search parameter file
    CoordinateOperatorParameters coordinate_operator_parameters(theory, dir);

    ConditionalIntersectionObserverParameters conditional_observer_parameters = build_conditional_intersection_observer_parameters();

    coordinate_operator_parameters.append_parameters(conditional_observer_parameters);
    coordinate_operator_parameters.write_to_file(dir);
}

void run_evolve_conditional_intersection_from_file()
{
    const std::string theory = "three_point_system";
    const std::string dir = "example_evolve_conditional_intersection";

    // Load existing fixed_point_search parameter file
    CoordinateOperatorParameters coordinate_operator_parameters(theory, dir);

    PathParameters path_parameters = coordinate_operator_parameters.get_path_parameters();
    Executer executer(path_parameters);
    executer.main(dir);
}

// Mode: "evolve_conditional"
void evolve_conditional_intersection()
{
    // Write necessary parameter files to file
    write_coordinate_operator_params_to_file("example_evolve_conditional_intersection", "evolve_with_conditional_intersection_observer");
    add_conditional_intersection_observer_parameters_to_coordinate_operator_to_file();
    add_evolve_on_condition_parameters_to_coordinate_operator_to_file<ConditionalIntersectionObserverParameters>("example_evolve_conditional_intersection");

    // Run
    run_evolve_conditional_intersection_from_file();
} */

#endif //PROGRAM_EVOLVE_CONDITIONALLY_HPP
