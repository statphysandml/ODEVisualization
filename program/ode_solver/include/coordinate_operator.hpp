//
// Created by lukas on 31.07.19.
//

#ifndef PROGRAM_COORDINATE_OPERATOR_HPP
#define PROGRAM_COORDINATE_OPERATOR_HPP

#include "util/header.hpp"
#include "util/dev_dat.hpp"
#include "util/helper_functions.hpp"
#include "util/frgvisualization_parameters.hpp"

#include "extern/thrust_functors.hpp"

#include "flow_equation_interface/flow_equation_system.hpp"
#include "flow_equation_interface/jacobian_equation.hpp"
#include "observers/evolution.hpp"



#include "Eigen/Dense"

#include <utility>

using json = nlohmann::json;


// Similar function as comput_vertex_velocities //ToDo: Move? currently not because of usage of eigen
std::vector< Eigen::MatrixXd* > compute_jacobian_elements(const DevDatC  &coordinates, JacobianWrapper * jacobian_equations);


// ToDo: It should also be possible to use this function without defining coordinates -> necessary??
class CoordinateOperatorParameters : public FRGVisualizationParameters {
public:
    explicit CoordinateOperatorParameters(const json params_, const PathParameters path_parameters_);

    // From file
    static CoordinateOperatorParameters from_file(const std::string theory,
            const std::string mode_type,
            const std::string dir,
            const std::string root_dir="/data/",
            const bool relative_path=true);

    static CoordinateOperatorParameters from_parameters(
            const std::string theory_,
            const std::vector <std::vector<double> > initial_coordinates={},
            const std::string mode_="evaluate", // default mode
            const std::string root_dir="/data/",
            const bool relative_path=true
    );

    template<typename ObserverParameters>
    void set_observer_params(ObserverParameters& observer_parameters)
    {
        append_parameters(observer_parameters, "observer");
    }

    void set_observer_params(std::string observer_name)
    {
        params["observer"] = observer_name;
    }

    json get_observer_params() const
    {
        return get_value_by_key<json>("observer");
    }

    void set_evolution_params(EvolutionParameters &evolution_parameters)
    {
        append_parameters(evolution_parameters);
    }

    json get_evolution_params() const
    {
        return get_value_by_key<json>("evolution");
    }

private:
    friend class CoordinateOperator;

    const uint dim;
    const cudaT k;

    std::vector< std::vector<double> > initial_coordinates;

    FlowEquationsWrapper * flow_equations;
    JacobianWrapper * jacobian_equations;
};


class CoordinateOperator
{
public:
    explicit CoordinateOperator(const CoordinateOperatorParameters &ep_) : ep(ep_)
    {
        if(ep.initial_coordinates.size() > 0)
            raw_coordinates = DevDatC(ep.initial_coordinates);
    }

    // Coordinates
    void set_raw_coordinates(const DevDatC coordinates);

    // Velocities
    void compute_velocities();

    // Jacobians and eigendata
    void compute_jacobians_and_eigendata();

    // Writes velocities, jacobians and eigendata to file
    void write_characteristics_to_file(const std::string dir) const;

    //[ Functions for evolution

    /* template<typename Observer=PseudoObserver> // ToDo! -> With variants?
    Observer load_observer_from_parameters()
    {
        observer = generate_observer<Observer>(ep.get_value_by_key("observer_parameters"));
    } */

    template<typename Observer=PseudoObserver>
    void evolve(Observer &observer, const std::string integration_type="equidistant_time_fixed_end", const cudaT start_t=0.0,
            const cudaT delta_t=0.01, const cudaT end_t=1.0, const uint number_of_observations=0,
            const uint observe_every_nth_step=0, const std::string step_size_type="constant",
            const std::string results_dir="", const uint max_steps_between_observations=500);

    template<typename Observer=PseudoObserver>
    void evolve_from_parameters(Observer &observer, const EvolutionParameters evolve_parameters);

    template<typename Observer=PseudoObserver>
    void evolve_from_file(Observer &observer);

    /* template<typename ConditionalObserverParameters>
    void evolve_on_condition(
            const ConditionalObserverParameters conditional_observer_parameters,
            const std::string results_dir="",
            const uint observe_every_nth_step = 10, const uint maximum_total_number_of_steps = 1000000);

    template<typename ConditionalObserverParameters>
    void evolve_on_condition_from_parameters(const CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalObserverParameters> evolve_on_condition_parameters);

    template<typename ConditionalObserverParameters>
    void evolve_on_condition_from_file(); */

    //]

    //[ Getter functions

    std::vector< std::vector<double> > get_initial_coordinates() const;

    DevDatC get_raw_coordinates() const;
    DevDatC get_raw_velocities() const;

    std::vector< Eigen::MatrixXd* > get_jacobians() const;

    Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType get_eigenvector(const int i) const;
    Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType get_eigenvalue(const int i) const;

    std::vector<std::vector< std::vector<cudaT>>> get_real_parts_of_eigenvectors() const;
    std::vector<std::vector<cudaT>> get_real_parts_of_eigenvalues() const;

    //]

    // Special functions

    std::vector<int> get_indices_with_saddle_point_characteristics() const;

private:
    const CoordinateOperatorParameters &ep;

    DevDatC raw_coordinates;
    DevDatC raw_velocities;

    // For the computation of jacobians
    std::vector< Eigen::MatrixXd* > jacobians;
    std::vector< Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType > eigenvectors;
    std::vector< Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType> eigenvalues;

    json vec_vec_to_json(const std::vector < std::vector<double> > data) const;
    json jacobians_to_json() const;

    json eigenvectors_to_json() const;
    json eigenvalues_to_json() const;
};

template<typename Observer>
void CoordinateOperator::evolve(Observer &observer, const std::string integration_type, const cudaT start_t,
                                const cudaT delta_t, const cudaT end_t, const uint number_of_observations,
                                const uint observe_every_nth_step, const std::string step_size_type,
                                const std::string results_dir, const uint max_steps_between_observations)
{
    EvolutionParameters evolution_parameters (json {{"integration_type", integration_type}, {"start_t", start_t}, {"delta_t", delta_t}, {"end_t", end_t},
                              {"number_of_observations", number_of_observations}, {"observe_every_nth_step", },
                              {"step_size_type", step_size_type}, {"results_dir", results_dir},
                              {"max_steps_between_observations", max_steps_between_observations},
                              {"conditional_observer_name", Observer::name()}});
    evolve_from_parameters(observer, evolution_parameters);
    /* if (results_dir != "") {
        std::string path = ep.get_absolute_path(ep.path_parameters.get_base_path(), ep.path_parameters.relative_path);
        Fileos fileos(path + results_dir + "trajectories.dat");
        auto observer = new TrackingObserver(fileos.get());
        Evolution<TrackingObserver> evaluator(ep.flow_equations, observer);
        print_range("Initial point", raw_coordinates.begin(), raw_coordinates.end());
        evaluator.evolve(raw_coordinates, start_t, end_t, delta_t);
        print_range("End point", raw_coordinates.begin(), raw_coordinates.end());
    } else {
        auto observer = new PseudoObserver();
        Evolution<PseudoObserver> evaluator(ep.flow_equations, observer);
        print_range("Initial point", raw_coordinates.begin(), raw_coordinates.end());
        evaluator.evolve(raw_coordinates, start_t, end_t, delta_t);
        print_range("End point", raw_coordinates.begin(), raw_coordinates.end());
    } */
}

template<typename Observer>
void CoordinateOperator::evolve_from_parameters(Observer &observer, const EvolutionParameters evolution_parameters)
{
    Evolution evolution(evolution_parameters, ep.flow_equations);
    print_range("Initial point", raw_coordinates.begin(), raw_coordinates.end());
    evolution.evolve(raw_coordinates, observer);
    print_range("End point", raw_coordinates.begin(), raw_coordinates.end());
}

// store observer_params on same level as evolution in coordinate operator params file.

template<typename Observer>
void CoordinateOperator::evolve_from_file(Observer &observer)
{
    auto evolve_params = ep.get_evolution_params();
    auto evolve_parameters = EvolutionParameters(evolve_params);

    evolve_from_parameters(observer, evolve_parameters);
}


/* template<typename ConditionalObserverParameters>
void CoordinateOperator::evolve_on_condition(const ConditionalObserverParameters conditional_observer_parameters,
                                              const std::string results_dir,
                                              const uint observe_every_nth_step, const uint maximum_total_number_of_steps)
{
    if(results_dir != "")
    {
        std::string path = ep.get_absolute_path(ep.path_parameters.get_base_path(), ep.path_parameters.relative_path);
        Fileos fileos(path + results_dir + "trajectories.dat");
        auto observer = new typename ConditionalObserverParameters::ConditionalObserver(ep.flow_equations, uint(raw_coordinates.size()), fileos.get(), conditional_observer_parameters);
        Evolution<typename ConditionalObserverParameters::ConditionalObserver> evaluator(ep.flow_equations, observer, observe_every_nth_step, maximum_total_number_of_steps);
        print_range("Initial point", raw_coordinates.begin(), raw_coordinates.end());
        evaluator.evolve_observer_based(raw_coordinates, 0.01);
        print_range("End point", raw_coordinates.begin(), raw_coordinates.end());
    } */
    /* else
    {
        auto observer = new typename ConditionalObserverParameters::ConditionalObserver(conditional_observer_parameters);
        Evolution<typename ConditionalObserverParameters::ConditionalObserver> evaluator(ep.flow_equations, observer, observe_every_nth_step, maximum_total_number_of_steps);
        print_range("Initial point", raw_coordinates.begin(), raw_coordinates.end());
        evaluator.evolve_observer_based(raw_coordinates, 0.01);
        print_range("End point", raw_coordinates.begin(), raw_coordinates.end());
    } */
/* }

template<typename ConditionalObserverParameters>
void CoordinateOperator::evolve_on_condition_from_parameters(const CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalObserverParameters> evolve_on_condition_parameters)
{ */
    /* auto conditional_observer_parameters = ep.get_value_by_key("conditional_observer");
    auto evolve_on_condition_parameters = ep.get_value_by_key("evolve_on_condition");
    auto params1 = ConditionalRangeObserverParameters(conditional_observer_parameters);
    auto params2 = CoordinateOperatorParameters::EvolveOnConditionParameters(evolve_on_condition_parameters);

    evolve_on_condition(results_dir, params1.boundary_lambda_ranges, params1.minimum_change_of_state,
                         params2.observe_every_nth_step,
                         params2.maximum_total_number_of_steps,
                         params1.minimum_delta_t, params1.maximum_flow_val); */
/* }

template<typename ConditionalObserverParameters>
void CoordinateOperator::evolve_on_condition_from_file()
{
    auto evolve_on_condition_parameters = ep.get_value_by_key<json>("evolve_on_condition");
    auto params = CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalObserverParameters>(evolve_on_condition_parameters);
    auto conditional_observer_params = ep.get_value_by_key<json>(params.conditional_oberserver_name);
    auto conditional_observer_parameters = ConditionalRangeObserverParameters(conditional_observer_params);

    evolve_on_condition(conditional_observer_parameters, params.results_dir,
                         params.observe_every_nth_step, params.maximum_total_number_of_steps);
} */

#endif //PROGRAM_COORDINATE_OPERATOR_HPP
