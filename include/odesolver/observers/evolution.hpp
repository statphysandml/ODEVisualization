//
// Created by lukas on 04.08.19.
//

#ifndef PROGRAM_EVOLUTION_HPP
#define PROGRAM_EVOLUTION_HPP

#include "pseudo_observer.hpp"
#include "tracking_observer.hpp"
// #include "conditional_range_observer.hpp"
// #include "conditional_intersection_observer.hpp"

#include <odesolver/boost/dev_dat_boost_integration.hpp>
#include <odesolver/boost/boost_integrators.hpp>

// ToDo: Latest Changes mid of april - most of the comments were recently added - handwritten notes with respect to the
// integrators exist, variables should already be complete for explizit implementation (i.e., the parameter class has been adpated

// Stepper types

typedef boost::numeric::odeint::runge_kutta_dopri5< odesolver::DevDatC, cudaT,  odesolver::DevDatC, cudaT > error_stepper;
typedef boost::numeric::odeint::runge_kutta4< odesolver::DevDatC, cudaT,  odesolver::DevDatC, cudaT > stepper;

// observer_types::variant_type;

// Generic function that tries to decode json to any type and
// returns true if the conversion succeeded. Otherwise it returns false.

/* bool aux_try_decode(const std::string observer_name, observer_variants::variant_type &variant)
{
    if(variant::name() == observer_name)
        return true;
    else
        return false;
} */

class Evolution : public param_helper::params::Parameters
{
public:
    enum StepSizeType {constant, adpative};
    static const std::map< std::string, StepSizeType> mode_resolver;

    Evolution(const json params, odesolver::flowequations::FlowEquationsWrapper * flow_equations_):
        Parameters(params),
        system(odesolver::flowequations::FlowEquationSystem(flow_equations_)),
        integration_type(get_entry<std::string>("integration_type")),
        start_t(get_entry<cudaT>("start_t")),
        delta_t(get_entry<cudaT>("delta_t")),
        end_t(get_entry<cudaT>("end_t")),
        number_of_observations(get_entry<uint>("number_of_observations")),
        observe_every_nth_step(get_entry<uint>("observe_every_nth_step")),
        step_size_type(mode_resolver.at(get_entry<std::string>("step_size_type"))),
        results_dir(get_entry<std::string>("results_dir")),
        max_steps_between_observations(get_entry<uint>("max_steps_between_observations"))
    {}

    // Evaluate at equidistant time steps and fixed end time
    Evolution(odesolver::flowequations::FlowEquationsWrapper * flow_equations_, const cudaT start_t_, const cudaT delta_t_, const cudaT end_t_,
                     const std::string step_size_type_="constant", const std::string results_dir_="",
                     const uint max_steps_between_observations_=500) :
            Evolution(json {{"integration_type", "equidistant_time_fixed_end"}, {"start_t", start_t_}, {"delta_t", delta_t_}, {"end_t", end_t_},
            {"number_of_observations", 0}, {"observe_every_nth_step", 0},
            {"step_size_type", step_size_type_}, {"results_dir", results_dir_},
            {"max_steps_between_observations", max_steps_between_observations_}}, flow_equations_)
    {}

    // Evaluate at equidistant time steps and a fixed number of observations
    Evolution(odesolver::flowequations::FlowEquationsWrapper * flow_equations_, const cudaT start_t_, const cudaT delta_t_, const uint number_of_observations_,
                     const std::string step_size_type_="constant", const std::string results_dir_="",
                     const uint max_steps_between_observations_=500) :
            Evolution(json {{"integration_type", "equidistant_time_fixed_n"}, {"start_t", start_t_}, {"delta_t", delta_t_}, {"end_t", 0.0},
            {"number_of_observations", number_of_observations_}, {"observe_every_nth_step", 0},
            {"step_size_type", step_size_type_}, {"results_dir", results_dir_},
            {"max_steps_between_observations", max_steps_between_observations_}}, flow_equations_)
    {}

    // Evaluate after at equidistant integration steps and fixed end time
    Evolution(odesolver::flowequations::FlowEquationsWrapper * flow_equations_, const uint observe_every_nth_step_, const cudaT start_t_, const cudaT delta_t_,
                     const cudaT end_t_, const std::string step_size_type_="constant", const std::string results_dir_="",
                     const uint max_steps_between_observations_=500) :
            Evolution(json {{"integration_type", "equidistant_integration_steps_fixed_end"}, {"start_t", start_t_}, {"delta_t", delta_t_}, {"end_t", end_t_},
            {"number_of_observations", 0}, {"observe_every_nth_step", observe_every_nth_step_},
            {"step_size_type", step_size_type_}, {"results_dir", results_dir_},
            {"max_steps_between_observations", max_steps_between_observations_}}, flow_equations_)
    {}

    // Evaluate after at equidistant integration steps and a fixed number of observations
    Evolution(odesolver::flowequations::FlowEquationsWrapper * flow_equations_, const uint observe_every_nth_step_, const cudaT start_t_, const cudaT delta_t_,
                     const uint number_of_observations_, const std::string step_size_type_="constant",
                     const std::string results_dir_="", const uint max_steps_between_observations_=500) :
            Evolution(json {{"integration_type", "equidistant_integration_steps_fixed_n"}, {"start_t", start_t_},
            {"delta_t", delta_t_}, {"end_t", 0.0}, {"number_of_observations", number_of_observations_},
            {"observe_every_nth_step", observe_every_nth_step_}, {"step_size_type", step_size_type_},
            {"results_dir", results_dir_}, {"max_steps_between_observations", max_steps_between_observations_}}, flow_equations_)
    {}

    /* These conditions manage if observer is invoked at every integrator step, at equidistant time steps or after a finite number of integrator steps
     * If observe_every_nth_step=1: use integrate_adaptive - the total number of observations depends on the used step_size_type and end_t,
     *     in the case of a constant step_size the number of resulting observations is (end_t - start_t)/delta_t and therefore equivalent to A
     * elif observe_at_equidistant_time_steps=finite: (observe_every_nth_step=0)
     *     if number_of_observations_!=0:
     *         use integrate_n_steps (the total number of observations is fixed in this case by number_of_observations
     *     elif end_t != 0.0: (A)
     *         use integrate_const (the total number of observations is fixed in this case by (end_t - start_t)/delta_t
     * elif observe_every_nth_step>1:
     *     use integrate_adpative - with an additional while loop - the total number of observations depends on the used step_size type and end_t,
     *         in the case of a constant step_size the number of resulting observations is (end_t - start_t)/delta_t / observe_every_nth_step */


    std::string name() const
    {
        return "evolution";
    }

        /* stepper
         * (store results_dir)
         * observer?
         * on condition? */

    // Evolution ends based on end_t or the given maximum number of steps
    template<typename Observer>
    void evolve(odesolver::DevDatC &initial_coordinates, Observer &observer)
    {
        /* ToDo: Check if this is a valid argument for another condition for adaptive integration:
         * Evolution of a 1D "particle" - The assumption is that an adaptive integration is only in this case
         * reasonable since a different step size involves different accuracies for different coordinates system.get_dim() == initial_coordinates.size() */
        /* switch (mode) {
            case constant: {
                std::cout << "Performing integration with constant step size" << std::endl;
                auto integrator = new ConstantIntegrator<constant_stepper, Observer>();
                constant_stepper rk;
                evolve_<constant_stepper>(rk, integrator, initial_coordinates, start_t, end_t, delta_t);
                delete integrator;
            }
            case adpative: {
                std::cout << "Performing integration with adaptive step size" << std::endl;
                boost::numeric::odeint::controlled_runge_kutta<error_stepper> stepper = boost::numeric::odeint::make_dense_output(
                        1.0e-6, 1.0e-6, 0.01, error_stepper())
                auto integrator = new AdaptiveIntegrator<controlled_stepper, Observer>();
                controlled_stepper rk;
                evolve_<controlled_stepper>(rk, integrator, initial_coordinates, start_t, end_t, delta_t);
                delete integrator;
            }
        } */
    }

    // Evolution is aborted based on a given condition that is provided by the observer
    template<typename Observer>
    void evolve_on_condition(odesolver::DevDatC &initial_coordinates, Observer &observer)
    {
        /* std::cout << "Performing constant integration based on observer" << std::endl;
        // ToDo: Why is this line defined?? integrator is not used!
        // auto integrator = new ConstantIntegrator<constant_stepper, Observer> ();
        constant_stepper rk;
        cudaT t = 0;
        uint number_of_steps = 0;
        while(!observer->check_for_out_of_system() && number_of_steps <= maximum_total_number_of_steps)
        {
            boost::numeric::odeint::integrate_n_steps(rk, system, initial_coordinates, t, delta_t, observe_every_nth_step);
            (*observer)(initial_coordinates, t);
            t += observe_every_nth_step * delta_t;
            number_of_steps += observe_every_nth_step;
        } */
    }

private:
    const cudaT start_t;
    const cudaT delta_t;
    const cudaT end_t; // corresponds to the maximum evolution time in case of a conditional observer
    const cudaT number_of_observations; // corresponds to the maximum number of observations in case of a conditional observer
    const uint observe_every_nth_step;
    const StepSizeType step_size_type; // Determines the used stepper type
    const std::string results_dir; // This refers to the results_dir of results of the evolution, not of the observer!!
    const uint max_steps_between_observations;
    const std::string integration_type;

    // ToDo: The dim variable can be removed at many points since present in flow equations, etc. (see also in other files)
    odesolver::flowequations::FlowEquationSystem system;

    template <typename StepperType, typename Observer>
    void evolve_(StepperType stepper, odesolver::boost_integrators::Integrator<StepperType, Observer> * integrator, odesolver::DevDatC &initial_coordinates, cudaT start_t=0.0, cudaT end_t=1.0, cudaT delta_t=0.01)
    {
        /* if(observer and observe_every_nth_step > 1)
        {
            // Note that the integrator is not used in this case
            cudaT t = start_t;
            uint number_of_steps = 0;
            while(t < end_t && number_of_steps <= maximum_total_number_of_steps)
            {
                // ToDo: Check if this is possible with contralled steppers, i.e. to have a variable delta t
                boost::numeric::odeint::integrate_n_steps(stepper, system, initial_coordinates, t, delta_t, observe_every_nth_step);
                (*observer)(initial_coordinates, t);
                t += observe_every_nth_step * delta_t;
                number_of_steps += observe_every_nth_step;
            }
        }
        else
            (*integrator) (stepper, system, initial_coordinates, start_t, end_t, delta_t, observer): */
    }
};

/* template<typename Observer=PseudoObserver>
Observer generate_observer<Observer>(json observer_params)
{
    return Observer()
} */

#endif //PROGRAM_EVOLUTION_HPP
