#ifndef PROGRAM_COORDINATE_OPERATOR_HPP
#define PROGRAM_COORDINATE_OPERATOR_HPP

#include <utility>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/json_conversions.hpp>
#include <odesolver/util/thrust_functors.hpp>
#include <odesolver/modes/ode_visualization.hpp>
#include <odesolver/observers/evolution.hpp>
#include <odesolver/eigen/jacobians.hpp>

using json = nlohmann::json;


namespace odesolver {
    namespace modes {
        class CoordinateOperator : public ODEVisualization
        {
        public:
            explicit CoordinateOperator(
                const json params={},
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr=nullptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root(),
                const std::vector<std::vector<double>> vecvec_coordinates={},
                const odesolver::DevDatC devdat_coordinates={}
            );
            
            // From parameters
            static CoordinateOperator generate(
                const odesolver::DevDatC devdat_coordinates,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr=nullptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            // From parameters
            static CoordinateOperator from_vecvec(
                const std::vector <std::vector<double> > vecvec_coordinates,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr=nullptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            // From file
            static CoordinateOperator from_file(
                const std::string rel_config_dir,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr,
                const std::string computation_parameters_path=param_helper::proj::project_root()
            );

            template<typename ObserverParameters>
            void set_observer_params(ObserverParameters& observer_parameters)
            {
                append_parameters(observer_parameters, "observer");
            }

            void set_observer_params(std::string observer_name)
            {
                params_["observer"] = observer_name;
            }

            json get_observer_params() const
            {
                return get_entry<json>("observer");
            }

            void set_evolution_params(Evolution &evolution)
            {
                append_parameters(evolution);
            }

            json get_evolution_params() const
            {
                return get_entry<json>("evolution");
            }

            // Coordinates

            void set_coordinates(const std::vector<std::vector<double>> coordinates);

            void set_coordinates(const odesolver::DevDatC coordinates);

            // Flow
            void compute_velocities();

            // Jacobians
            void compute_jacobians();

            // Writes velocities, jacobians and eigendata to file
            void write_characteristics_to_file(const std::string rel_dir) const;

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
            void evolve_from_parameters(Observer &observer, const Evolution evolve_parameters);

            template<typename Observer=PseudoObserver>
            void evolve_from_file(Observer &observer);

            /* template<typename ConditionalObserverParameters>
            void evolve_on_condition(
                    const ConditionalObserverParameters conditional_observer_parameters,
                    const std::string results_dir="",
                    const uint observe_every_nth_step = 10, const uint maximum_total_number_of_steps = 1000000);

            template<typename ConditionalObserverParameters>
            void evolve_on_condition_from_parameters(const CoordinateOperator::EvolveOnConditionParameters<ConditionalObserverParameters> evolve_on_condition_parameters);

            template<typename ConditionalObserverParameters>
            void evolve_on_condition_from_file(); */

            //]

            //[ Getter functions

            // std::vector<std::vector<double>> get_initial_coordinates() const;

            odesolver::DevDatC get_coordinates() const;
            odesolver::DevDatC get_velocities() const;
            odesolver::DevDatC get_jacobian_elements() const;

            odesolver::modes::Jacobians get_jacobians() const;

            //]

        private:
            const uint dim_;

            odesolver::DevDatC coordinates_;
            odesolver::DevDatC velocities_;
            odesolver::DevDatC jacobian_elements_;
            odesolver::modes::Jacobians jacobians_;
        };

        template<typename Observer>
        void CoordinateOperator::evolve(Observer &observer, const std::string integration_type, const cudaT start_t,
                                        const cudaT delta_t, const cudaT end_t, const uint number_of_observations,
                                        const uint observe_every_nth_step, const std::string step_size_type,
                                        const std::string results_dir, const uint max_steps_between_observations)
        {
            Evolution evolution_parameters (json {{"integration_type", integration_type}, {"start_t", start_t}, {"delta_t", delta_t}, {"end_t", end_t},
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
                print_range("Initial point", coordinates.begin(), coordinates.end());
                evaluator.evolve(coordinates, start_t, end_t, delta_t);
                print_range("End point", coordinates.begin(), coordinates.end());
            } else {
                auto observer = new PseudoObserver();
                Evolution<PseudoObserver> evaluator(ep.flow_equations, observer);
                print_range("Initial point", coordinates.begin(), coordinates.end());
                evaluator.evolve(coordinates, start_t, end_t, delta_t);
                print_range("End point", coordinates.begin(), coordinates.end());
            } */
        }

        template<typename Observer>
        void CoordinateOperator::evolve_from_parameters(Observer &observer, const Evolution evolution_parameters)
        {
            // ### Recently Todo
            /* Evolution evolution(evolution_parameters, ep.flow_equations);
            print_range("Initial point", coordinates.begin(), coordinates.end());
            evolution.evolve(coordinates, observer);
            print_range("End point", coordinates.begin(), coordinates.end()); */
        }

        // store observer_params on same level as evolution in coordinate operator params file.

        template<typename Observer>
        void CoordinateOperator::evolve_from_file(Observer &observer)
        {
            auto evolve_params = get_evolution_params();
            auto evolve_parameters = Evolution(evolve_params);

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
                auto observer = new typename ConditionalObserverParameters::ConditionalObserver(ep.flow_equations, uint(coordinates.size()), fileos.get(), conditional_observer_parameters);
                Evolution<typename ConditionalObserverParameters::ConditionalObserver> evaluator(ep.flow_equations, observer, observe_every_nth_step, maximum_total_number_of_steps);
                print_range("Initial point", coordinates.begin(), coordinates.end());
                evaluator.evolve_observer_based(coordinates, 0.01);
                print_range("End point", coordinates.begin(), coordinates.end());
            } */
            /* else
            {
                auto observer = new typename ConditionalObserverParameters::ConditionalObserver(conditional_observer_parameters);
                Evolution<typename ConditionalObserverParameters::ConditionalObserver> evaluator(ep.flow_equations, observer, observe_every_nth_step, maximum_total_number_of_steps);
                print_range("Initial point", coordinates.begin(), coordinates.end());
                evaluator.evolve_observer_based(coordinates, 0.01);
                print_range("End point", coordinates.begin(), coordinates.end());
            } */
        /* }

        template<typename ConditionalObserverParameters>
        void CoordinateOperator::evolve_on_condition_from_parameters(const CoordinateOperator::EvolveOnConditionParameters<ConditionalObserverParameters> evolve_on_condition_parameters)
        { */
            /* auto conditional_observer_parameters = ep.get_value_by_key("conditional_observer");
            auto evolve_on_condition_parameters = ep.get_value_by_key("evolve_on_condition");
            auto params1 = ConditionalRangeObserverParameters(conditional_observer_parameters);
            auto params2 = CoordinateOperator::EvolveOnConditionParameters(evolve_on_condition_parameters);

            evolve_on_condition(results_dir, params1.boundary_variable_ranges, params1.minimum_change_of_state,
                                params2.observe_every_nth_step,
                                params2.maximum_total_number_of_steps,
                                params1.minimum_delta_t, params1.maximum_flow_val); */
        /* }

        template<typename ConditionalObserverParameters>
        void CoordinateOperator::evolve_on_condition_from_file()
        {
            auto evolve_on_condition_parameters = ep.get_value_by_key<json>("evolve_on_condition");
            auto params = CoordinateOperator::EvolveOnConditionParameters<ConditionalObserverParameters>(evolve_on_condition_parameters);
            auto conditional_observer_params = ep.get_value_by_key<json>(params.conditional_oberserver_name);
            auto conditional_observer_parameters = ConditionalRangeObserverParameters(conditional_observer_params);

            evolve_on_condition(conditional_observer_parameters, params.results_dir,
                                params.observe_every_nth_step, params.maximum_total_number_of_steps);
        } */
    }
}

#endif //PROGRAM_COORDINATE_OPERATOR_HPP
