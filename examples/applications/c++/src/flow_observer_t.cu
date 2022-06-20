#include "../include/flow_observer_t.hpp"

void no_change_t()
{
    odesolver::DevDatC sampled_coordinates = gen_normal_devdat(3, 10);

    auto no_change_condition = odesolver::evolution::NoChange::generate({0.1, 0.0001, 0.02});
    no_change_condition.initialize(sampled_coordinates, 0.0);
    // no_change_condition(sampled_coordinates, 0.1);
    auto condition = no_change_condition.valid_coordinates_mask();
    
    sampled_coordinates.print_dim_by_dim();
    print_range("Condition", condition.begin(), condition.end());

    sampled_coordinates.set_nth_element(1, {1.0, 0.2, 0.1});
    sampled_coordinates.set_nth_element(2, {9.0}, 1);
    sampled_coordinates.print_dim_by_dim();

    no_change_condition(sampled_coordinates, 0.2);

    condition = no_change_condition.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());
}

void divergent_flow_t()
{
    odesolver::DevDatC sampled_coordinates = gen_normal_devdat(3, 10);

    auto flow_equations_ptr = odesolver::flowequations::generate_flow_equations<LorentzAttractorFlowEquations>(0);

    auto flow = odesolver::flowequations::compute_flow(sampled_coordinates, flow_equations_ptr.get());

    flow.print_dim_by_dim();

    auto divergent_flow_condition = odesolver::evolution::DivergentFlow::generate(flow_equations_ptr, 30.0);
    divergent_flow_condition.initialize(sampled_coordinates, 0.0);

    divergent_flow_condition(sampled_coordinates, 0.1);

    auto condition = divergent_flow_condition.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());
}

void out_of_range_t()
{
    odesolver::DevDatC sampled_coordinates = gen_normal_devdat(3, 10);
    sampled_coordinates.print_dim_by_dim();

    const std::vector<std::pair<cudaT, cudaT>> variable_ranges = std::vector <std::pair<cudaT, cudaT>>{
        std::pair<cudaT, cudaT> (-12.0, 12.0), std::pair<cudaT, cudaT> (-12.0, 12.0), std::pair<cudaT, cudaT> (-1.0, 31.0)};

    auto out_of_range_condition = odesolver::evolution::OutOfRangeCondition::generate(variable_ranges);
    out_of_range_condition.initialize(sampled_coordinates, 0.0);

    out_of_range_condition(sampled_coordinates, 0.1);

    auto condition = out_of_range_condition.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());
    
    sampled_coordinates.set_nth_element(1, {1.0, 0.2, -1.1});
    sampled_coordinates.set_nth_element(2, {29.0}, 1);
    sampled_coordinates.print_dim_by_dim();

    out_of_range_condition(sampled_coordinates, 0.1);
    
    condition = out_of_range_condition.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());
}

void out_of_range_t2()
{
    odesolver::DevDatC sampled_coordinates = gen_normal_devdat(3, 10);
    sampled_coordinates.print_dim_by_dim();

    const std::vector<std::pair<cudaT, cudaT>> variable_ranges = std::vector <std::pair<cudaT, cudaT>>{
        std::pair<cudaT, cudaT> (-12.0, 12.0), std::pair<cudaT, cudaT> (-1.0, 31.0)};

    auto out_of_range_condition = odesolver::evolution::OutOfRangeCondition::generate(variable_ranges, {1, 2});
    out_of_range_condition.initialize(sampled_coordinates, 0.0);

    out_of_range_condition(sampled_coordinates, 0.1);

    auto condition = out_of_range_condition.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());
    
    sampled_coordinates.set_nth_element(1, {1.0, 0.2, -1.1});
    sampled_coordinates.set_nth_element(2, {29.0}, 1);
    sampled_coordinates.print_dim_by_dim();

    out_of_range_condition(sampled_coordinates, 0.1);
    
    condition = out_of_range_condition.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());
}

void trajectory_observer_t()
{
    odesolver::DevDatC sampled_coordinates = gen_normal_devdat(3, 10);
    sampled_coordinates.print_dim_by_dim();

    auto trajectory_observer = odesolver::evolution::TrajectoryObserver::generate("test.txt");
    trajectory_observer.initialize(sampled_coordinates, 0.0);

    trajectory_observer(sampled_coordinates, 0.1);
    
    sampled_coordinates.set_nth_element(1, {1.0, 0.2, -1.1});
    sampled_coordinates.set_nth_element(2, {29.0}, 1);
    
    trajectory_observer(sampled_coordinates, 0.2);
}

void intersection_observer_t()
{
    odesolver::DevDatC coordinates(4, 10, 0.0);
    coordinates.print_dim_by_dim();

    auto intersection = odesolver::evolution::Intersection::generate(
        {0.1, 0.2}, // vicinity_distances
        {0.04, 0.03}, // fixed_variables
        {1, 2}, // fixed_variable_indices
        true // remember_interactions
    );
    intersection.initialize(coordinates, 0.0);

    intersection(coordinates, 0.1);

    coordinates = odesolver::DevDatC(4, 10, -1.0);

    coordinates.set_nth_element(3, {0.05, 0.05, 0.05, 0.05});
    coordinates.set_nth_element(4, {0.05, -0.02, 0.05, 0.05});
    coordinates.set_nth_element(6, {0.05, 0.02, 0.02, 0.05});

    coordinates.print_dim_by_dim();

    intersection(coordinates, 0.2);

    auto intersection_counter = intersection.intersection_counter();
    for(const auto &c : intersection_counter)
        std::cout << c << " ";
    std::cout << std::endl;

    auto intersection_types = intersection.detected_intersection_types();
    for(const auto &c : intersection_types)
        std::cout << c << " ";
    std::cout << std::endl;

    auto intersections = intersection.detected_intersections();
    for(const auto &coor : intersections)
    {
        for(const auto &elem : coor)
            std::cout << elem << ", ";
        std::cout << std::endl;
    }
    
}

void evolution_observer_t()
{
    odesolver::DevDatC sampled_coordinates = gen_normal_devdat(3, 10);

    auto flow_equations_ptr = odesolver::flowequations::generate_flow_equations<LorentzAttractorFlowEquations>(0);

    std::shared_ptr<odesolver::evolution::FlowObserver> no_change_condition_ptr = std::make_shared<odesolver::evolution::NoChange>(odesolver::evolution::NoChange::generate({0.1, 0.0001, 0.02}));

    std::shared_ptr<odesolver::evolution::FlowObserver> divergent_flow_condition_ptr = std::make_shared<odesolver::evolution::DivergentFlow>(odesolver::evolution::DivergentFlow::generate(flow_equations_ptr, 30.0));

    // mu, lambda, g
    const std::vector<std::pair<cudaT, cudaT>> variable_ranges = std::vector <std::pair<cudaT, cudaT>>{
        std::pair<cudaT, cudaT> (-12.0, 12.0), std::pair<cudaT, cudaT> (-1.0, 31.0)};

    std::shared_ptr<odesolver::evolution::FlowObserver> out_of_range_condition_ptr = std::make_shared<odesolver::evolution::OutOfRangeCondition>(odesolver::evolution::OutOfRangeCondition::generate(variable_ranges, {1, 2}));

    auto evolution_observer = odesolver::evolution::EvolutionObserver::generate({no_change_condition_ptr, divergent_flow_condition_ptr, out_of_range_condition_ptr});
    evolution_observer.initialize(sampled_coordinates, 0.0);

    auto flow = odesolver::flowequations::compute_flow(sampled_coordinates, flow_equations_ptr.get());
    flow.print_dim_by_dim();

    auto condition = evolution_observer.valid_coordinates_mask();
    
    sampled_coordinates.print_dim_by_dim();
    print_range("Condition", condition.begin(), condition.end());
    std::cout << "n valid coordinates: " << evolution_observer.n_valid_coordinates() << std::endl;
    
    evolution_observer(sampled_coordinates, 0.2);

    sampled_coordinates.set_nth_element(1, {1.0, 0.2, 0.1});
    sampled_coordinates.set_nth_element(2, {9.0}, 1);
    sampled_coordinates.set_nth_element(4, {29.0}, 1);
    sampled_coordinates.print_dim_by_dim();

    evolution_observer(sampled_coordinates, 0.2);

    condition = evolution_observer.valid_coordinates_mask();
    print_range("Condition", condition.begin(), condition.end());

    std::cout << "n valid coordinates: " << evolution_observer.n_valid_coordinates() << std::endl;
}