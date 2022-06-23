#include "../include/separatrizes_t.hpp"


// ToDo: FlowEquation not really needed in this setup in separatirzes


void separatrizes_t()
{
    // Separatrizes
    auto separatrizes = odesolver::modes::Separatrizes::generate(
        3, // N_per_eigen_dim
        {0.01, 0.01, 0.01}, // shift_per_dim
        10000, // n_max_steps
        flowequations::generate_flow_equations<LorentzAttractorFlowEquations>(0),
        flowequations::generate_jacobian_equations<LorentzAttractorJacobianEquations>(0)
    );

    // Building the different observers
    auto flow_equations_ptr = flowequations::generate_flow_equations<LorentzAttractorFlowEquations>(0);

    std::shared_ptr<odesolver::evolution::NoChange> no_change_condition_ptr = std::make_shared<odesolver::evolution::NoChange>(odesolver::evolution::NoChange::generate({1e-6, 1e-6, 1e-6}));

    std::shared_ptr<odesolver::evolution::DivergentFlow> divergent_flow_condition_ptr = std::make_shared<odesolver::evolution::DivergentFlow>(odesolver::evolution::DivergentFlow::generate(flow_equations_ptr, 1e6));

    // mu, lambda, g
    const auto variable_ranges = std::vector <std::pair<cudaT, cudaT>>{std::pair<cudaT, cudaT>(-12.0, 12.0), std::pair<cudaT, cudaT>(-12.0, 12.0), std::pair<cudaT, cudaT>(-50.0, 50.0)};

    std::shared_ptr<odesolver::evolution::OutOfRangeCondition> out_of_range_condition_ptr = std::make_shared<odesolver::evolution::OutOfRangeCondition>(odesolver::evolution::OutOfRangeCondition::generate(variable_ranges));
    
    std::shared_ptr<odesolver::evolution::Intersection> intersection_ptr = std::make_shared<odesolver::evolution::Intersection>(odesolver::evolution::Intersection::generate(
        {0.1, 0.1}, // vicinity_distances
        {27.0}, // fixed_variables
        {2}, // fixed_variable_indices
        true // remember_interactions
    ));

    // Final Observer
    auto evolution_observer = odesolver::evolution::EvolutionObserver::generate({no_change_condition_ptr, divergent_flow_condition_ptr, out_of_range_condition_ptr, intersection_ptr});

    // Evolution
    auto evolution = odesolver::modes::Evolution::generate(flow_equations_ptr);
    // Stepper
    odesolver::evolution::stepper::RungaKutta4 stepper;
    
    // Fixed points
    double beta = 8.0/3.0;
    double rho = 28.0;
    std::vector<std::vector<cudaT>> fixed_points_ {
        {0.0, 0.0, 0.0},
        {std::sqrt(beta * (rho - 1)), std::sqrt(beta * (rho - 1)), rho - 1},
        {-std::sqrt(beta * (rho - 1)), -std::sqrt(beta * (rho - 1)), rho - 1}
    };

    devdat::DevDatC fixed_points(fixed_points_);

    separatrizes.eval(fixed_points, 0.001, evolution, stepper, evolution_observer);

    auto condition = evolution_observer.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());

    std::cout << "n valid coordinates: " << evolution_observer.n_valid_coordinates() << std::endl;

    /* auto intersection_counter = intersection_ptr->intersection_counter();
    for(const auto &c : intersection_counter)
        std::cout << c << " ";
    std::cout << std::endl;

    auto intersection_types = intersection_ptr->detected_intersection_types();
    for(const auto &c : intersection_types)
        std::cout << c << " ";
    std::cout << std::endl;

    auto intersections = intersection_ptr->detected_intersections();
    for(const auto &coor : intersections)
    {
        for(const auto &elem : coor)
            std::cout << elem << ", ";
        std::cout << std::endl;
    }*/
}

