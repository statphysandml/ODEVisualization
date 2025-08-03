#include "../include/evolution_t.hpp"


void evolution_t()
{
    auto flow_equations_ptr = flowequations::generate_flow_equations<LorentzAttractorFlowEquations>(0);

    devdat::DevDatC sampled_coordinates = gen_normal_devdat(3, 10);

    auto evolution = odesolver::modes::Evolution::generate(flow_equations_ptr);

    odesolver::evolution::stepper::RungaKutta4 stepper;

    sampled_coordinates.print_dim_by_dim();

    evolution.evolve_n_steps(stepper, sampled_coordinates, 0.0, 0.05, 1);

    sampled_coordinates.print_dim_by_dim();
}

void conditional_range_observer_t()
{
    devdat::DevDatC sampled_coordinates = gen_normal_devdat(3, 10);

    auto flow_equations_ptr = flowequations::generate_flow_equations<LorentzAttractorFlowEquations>(0);

    std::shared_ptr<odesolver::evolution::FlowObserver> no_change_condition_ptr = std::make_shared<odesolver::evolution::NoChange>(odesolver::evolution::NoChange::generate({0.0001, 0.0001, 0.0001}));

    std::shared_ptr<odesolver::evolution::FlowObserver> divergent_flow_condition_ptr = std::make_shared<odesolver::evolution::DivergentFlow>(odesolver::evolution::DivergentFlow::generate(flow_equations_ptr, 10e10));

    // mu, lambda, g
    const std::vector<std::pair<cudaT, cudaT>> variable_ranges = std::vector <std::pair<cudaT, cudaT>>{
        std::pair<cudaT, cudaT> (-12.0, 12.0), std::pair<cudaT, cudaT> (-31.0, 31.0)};

    std::shared_ptr<odesolver::evolution::FlowObserver> out_of_range_condition_ptr = std::make_shared<odesolver::evolution::OutOfRangeCondition>(odesolver::evolution::OutOfRangeCondition::generate(variable_ranges, {1, 2}));

    auto evolution_observer = odesolver::evolution::EvolutionObserver::generate({no_change_condition_ptr, divergent_flow_condition_ptr, out_of_range_condition_ptr});
    evolution_observer.initialize(sampled_coordinates, 0.0);

    auto evolution = odesolver::modes::Evolution::generate(flow_equations_ptr);

    odesolver::evolution::stepper::RungaKutta4 stepper;

    sampled_coordinates.print_dim_by_dim();

    evolution.evolve_n_steps(stepper, sampled_coordinates, 0.0, 0.001, 1, evolution_observer);

    sampled_coordinates.print_dim_by_dim();

    auto condition = evolution_observer.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());
    std::cout << "n valid coordinates: " << evolution_observer.n_valid_coordinates() << std::endl;
}