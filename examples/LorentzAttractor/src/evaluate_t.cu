#include "../include/evaluate_t.hpp"

odesolver::modes::CoordinateOperator build_coordinate_operator_parameters()
{
    double beta = 8.0/3.0;
    double rho = 28.0;

    std::vector < std::vector<cudaT> > coordinates {
        {0.0, 0.0, 0.0},
        {std::sqrt(beta * (rho - 1)), std::sqrt(beta * (rho - 1)), rho - 1},
        {-std::sqrt(beta * (rho - 1)), -std::sqrt(beta * (rho - 1)), rho - 1}
    };

    odesolver::modes::CoordinateOperator evaluator = odesolver::modes::CoordinateOperator::from_vecvec(
        coordinates,
        odesolver::flowequations::generate_flow_equations<LorentzAttractorFlowEquations>(0),
        odesolver::flowequations::generate_jacobian_equations<LorentzAttractorJacobianEquations>(0)
    );
    return evaluator;
}

void write_coordinate_operator_params_to_file(const std::string rel_dir)
{
    odesolver::modes::CoordinateOperator evaluator = build_coordinate_operator_parameters();
    evaluator.write_configs_to_file(rel_dir);
}

// Mode: "jacobian"
void run_evaluate_velocities_and_jacobians_from_file()
{
    const std::string rel_dir = "data/example_evaluate_velocities_and_jacobians";

    // Load existing parameter file
    odesolver::modes::CoordinateOperator evaluator = odesolver::modes::CoordinateOperator::from_file(
        rel_dir,
        odesolver::flowequations::generate_flow_equations<LorentzAttractorFlowEquations>(0),
        odesolver::flowequations::generate_jacobian_equations<LorentzAttractorJacobianEquations>(0)
    );

    double beta = 8.0/3.0;
    double rho = 28.0;

    std::vector < std::vector<cudaT> > coordinates {
        {0.0, 0.0, 0.0},
        {std::sqrt(beta * (rho - 1)), std::sqrt(beta * (rho - 1)), rho - 1},
        {-std::sqrt(beta * (rho - 1)), -std::sqrt(beta * (rho - 1)), rho - 1}
    };

    evaluator.set_coordinates(coordinates);

    evaluator.compute_velocities();
    evaluator.compute_jacobians();

    evaluator.write_characteristics_to_file(rel_dir);

    // (In principle four options)
    // Executer::exec_evaluate(evaluator, dir);
}

void evaluate_velocities_and_jacobians()
{
    // Write necessary parameter files to file
    write_coordinate_operator_params_to_file("data/example_evaluate_velocities_and_jacobians");

    // Run
    run_evaluate_velocities_and_jacobians_from_file();
}

/* void evolve_a_hypercube()
{
    // ToDo! Evolve a set of hypercubes
} */
