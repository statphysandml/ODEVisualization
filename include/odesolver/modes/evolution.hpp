#ifndef PROGRAM_EVOLUTION_HPP
#define PROGRAM_EVOLUTION_HPP

#include <param_helper/params.hpp>

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <devdat/util/json_conversions.hpp>
#include <odesolver/util/monitor.hpp>
#include <flowequations/flow_equation_system.hpp>
#include <odesolver/boost/dev_dat_boost_integration.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <odesolver/evolution/evolution_observer.hpp>
#include <odesolver/modes/ode_visualization.hpp>


namespace odesolver {
    namespace modes {

        struct Evolution : public ODEVisualization
        {
            // From config
            explicit Evolution(
                const json params,
                std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr
            );

            // From parameters
            static Evolution generate(
                std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr
            );
            
            // From file
            static Evolution from_file(
                const std::string rel_config_dir,
                std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<flowequations::JacobianEquationsWrapper> jacobians_ptr
            );

            template<typename StepperClass>
            void evolve_const(StepperClass &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT end_t, const cudaT delta_t)
            {
                integrate_const(stepper.stepper_, coordinates, start_t, end_t, delta_t);
            }

            template<typename StepperClass, typename Observer>
            void evolve_const(StepperClass &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT end_t, const cudaT delta_t, Observer &observer, bool equidistant_time_observations=true, const uint observe_every_ith_time_step=1)
            {
                // Constant step size
                if(stepper.concept() == "stepper" || stepper.concept() == "error_stepper")
                {
                    // Call observer after each dt time step and integrate with dt
                    if(observe_every_ith_time_step == 1)
                    {
                        integrate_const(stepper.stepper_, coordinates, start_t, end_t, delta_t, observer);
                    }
                    // Call observer after observe_every_ith_time_step * dt and integrate with dt
                    else
                    {
                        uint n = uint((end_t - start_t) / delta_t);
                        auto t_n_steps = integrate_n_steps_observe_every_ith_step(stepper, coordinates, start_t, delta_t, n, observer, observe_every_ith_time_step);
                        if(observer.valid_coordinates())
                            integrate_const(stepper.stepper_, coordinates, t_n_steps, end_t, delta_t);
                    }
                }
                // Adaptive step size
                else
                {
                    // Call observer after dt and integrate with adaptive step size
                    if(equidistant_time_observations)
                    {
                        cudaT dt = delta_t;
                        if(observe_every_ith_time_step > 1)
                            dt = observe_every_ith_time_step * delta_t;
                        integrate_const(stepper.stepper_, coordinates, start_t, end_t, dt, observer);
                    }
                    else
                    {
                        // Call observer after each time step with variable dt and integrate with adaptive step size
                        if(observe_every_ith_time_step==1)
                        {
                            integrate_adaptive(stepper.stepper_, coordinates, start_t, end_t, delta_t, observer);
                        }
                        // Call observer after every observe_every_ith_time_step time steps with variable dt and integrate with adaptive step size
                        else
                        {
                            std::cout << "not implemented" << std::endl;
                            // -> can probably be solved easily with a four loop and stepper.stepper_.do_step()??
                            // !!special!! integrate_adaptive, controlled or dense output stepper with sufficiently large end_t
                        }
                    }
                }
            }

            template<typename StepperClass>
            void evolve_n_steps(StepperClass &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT delta_t, const uint n)
            {
                integrate_n_steps(stepper.stepper_, coordinates, start_t, delta_t, n);
            }

            template<typename StepperClass, typename Observer>
            void evolve_n_steps(StepperClass &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT delta_t, const uint n, Observer &observer, bool equidistant_time_observations=true, const uint observe_every_ith_time_step=1)
            {
                // Constant step size
                if(stepper.concept() == "stepper" || stepper.concept() == "error_stepper")
                {
                    // Call observer after each dt time step and integrate with dt
                    if(observe_every_ith_time_step == 1)
                    {
                        integrate_n_steps(stepper.stepper_, coordinates, start_t, delta_t, n, observer);
                    }
                    // Call observer after observe_every_ith_time_step * dt and integrate with dt
                    else
                    {
                        integrate_n_steps_observe_every_ith_step(stepper, coordinates, start_t, delta_t, n, observer, observe_every_ith_time_step);
                    }
                }
                // Adaptive step size
                else
                {
                    // Call observer after dt and integrate with adaptive step size
                    if(equidistant_time_observations)
                    {
                        cudaT dt = delta_t;
                        if(observe_every_ith_time_step > 1)
                            dt = observe_every_ith_time_step * delta_t;
                        integrate_n_steps(stepper.stepper_, coordinates, start_t, dt, n, observer);
                    }
                    else
                    {
                        // Call observer after each time step with variable dt and integrate with adaptive step size
                        if(observe_every_ith_time_step==1)
                        {
                            std::cout << "not implemented" << std::endl;
                            // -> can probably be solved easily with a four loop and stepper.stepper_.do_step()??
                            // !!special!! integrate_adaptive, controlled or dense output stepper with sufficiently large end_t
                        }
                        // Call observer after every observe_every_ith_time_step time steps with variable dt and integrate with adaptive step size
                        else
                        {
                            integrate_n_steps_observe_every_ith_step(stepper, coordinates, start_t, delta_t, n, observer, observe_every_ith_time_step);
                        }
                    }
                }
            }

            template<typename StepperClass, typename Observer>
            cudaT integrate_n_steps_observe_every_ith_step(StepperClass &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT delta_t, const uint n, Observer &observer, const uint observe_every_ith_time_step=1)
            {
                cudaT t0 = start_t;
                uint total_n = 0;
                while((total_n < n) and observer.valid_coordinates())
                {
                    observer(coordinates, t0);
                    t0 += integrate_n_steps(stepper.stepper_, coordinates, t0, delta_t, observe_every_ith_time_step);
                    total_n += observe_every_ith_time_step;
                }
                if(observer.valid_coordinates())
                    observer(coordinates, t0);
                return t0;
            }

            template<typename Stepper>
            void integrate_const(Stepper &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT end_t, const cudaT delta_t)
            {
                boost::numeric::odeint::integrate_const(stepper, flow_equations_system_, coordinates, start_t, end_t, delta_t);
            }

            template<typename Stepper, typename Observer>
            void integrate_const(Stepper &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT end_t, const cudaT delta_t, Observer &observer)
            {
                try
                {
                    boost::numeric::odeint::integrate_const(stepper, flow_equations_system_, coordinates, start_t, end_t, delta_t, observer);
                }
                catch (odesolver::evolution::StopEvolutionException& e)
                {
                    std::cout << "Integration stopped at t=" << e.get_end_time()<< " since based on the " << e.get_reason() << " observer no valid coordinates are left." << std::endl;
                }
            }

            template<typename Stepper>
            cudaT integrate_n_steps(Stepper &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT delta_t, const uint n)
            {
                return boost::numeric::odeint::integrate_n_steps(stepper, flow_equations_system_, coordinates, start_t, delta_t, n);
            }

            template<typename Stepper, typename Observer>
            cudaT integrate_n_steps(Stepper &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT delta_t, const uint n, Observer &observer)
            {
                try
                {
                    return boost::numeric::odeint::integrate_n_steps(stepper, flow_equations_system_, coordinates, start_t, delta_t, n, observer);
                }
                catch (odesolver::evolution::StopEvolutionException& e)
                {
                    std::cout << "Integration stopped at t=" << e.get_end_time()<< " since based on the " << e.get_reason() << " observer no valid coordinates are left." << std::endl;
                    return e.get_end_time();
                }
            }

            template<typename Stepper>
            void integrate_adaptive(Stepper &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT end_t, const cudaT delta_t)
            {
                boost::numeric::odeint::integrate_adaptive(stepper, flow_equations_system_, coordinates, start_t, end_t, delta_t);
            }

            template<typename Stepper, typename Observer>
            void integrate_adaptive(Stepper &stepper, devdat::DevDatC &coordinates, const cudaT start_t, const cudaT end_t, const cudaT delta_t, Observer &observer)
            {
                try
                {
                    boost::numeric::odeint::integrate_adaptive(stepper, flow_equations_system_, coordinates, start_t, end_t, delta_t, observer);
                }
                catch (odesolver::evolution::StopEvolutionException& e)
                {
                    std::cout << "Integration stopped at t=" << e.get_end_time()<< " since based on the " << e.get_reason() << " observer no valid coordinates are left." << std::endl;
                }
            }

            flowequations::FlowEquationSystem flow_equations_system_;
        };
    }
}

#endif //PROGRAM_EVOLUTION_HPP
