#ifndef PROGRAM_BOOST_INTEGRATORS_HPP
#define PROGRAM_BOOST_INTEGRATORS_HPP


namespace odesolver {
    namespace boost_integrators {
        template <typename StepperType, typename Observer>
        struct Integrator
        {
            virtual void operator() (StepperType &stepper, odesolver::flowequations::FlowEquationSystem &system, odesolver::DevDatC &initial_coordinates, cudaT &start_t, cudaT &end_t, cudaT &delta_t, Observer &observer) = 0;
        };


        // For integrations with observer calls at each step
        template <typename StepperType, typename Observer>
        struct AdaptiveIntegrator : public Integrator<StepperType, Observer>
        {
            void operator() (StepperType &stepper, odesolver::flowequations::FlowEquationSystem &system, odesolver::DevDatC &initial_coordinates, cudaT &start_t, cudaT &end_t, cudaT &delta_t, Observer observer) override
            {
                if(observer.name() != "psuedo_observer")
                    boost::numeric::odeint::integrate_adaptive(stepper, system, initial_coordinates, start_t, end_t, delta_t, observer);
                else
                    boost::numeric::odeint::integrate_adaptive(stepper, system, initial_coordinates, start_t, end_t, delta_t);
            }
        };

        // For integrations with observer calls at equidistant time steps (const is referred to the given initial delta_t!)
        template <typename StepperType, typename Observer>
        struct ConstantIntegrator : public Integrator<StepperType, Observer>
        {
            void operator() (StepperType &stepper, odesolver::flowequations::FlowEquationSystem &system, odesolver::DevDatC &initial_coordinates, cudaT &start_t, cudaT &end_t, cudaT &delta_t, Observer &observer) override
            {
                if(observer.name() != "psuedo_observer")
                    boost::numeric::odeint::integrate_const(stepper, system, initial_coordinates, start_t, end_t, delta_t, observer);
                else
                    boost::numeric::odeint::integrate_const(stepper, system, initial_coordinates, start_t, end_t, delta_t);
            }
        };
    }
}

#endif //PROGRAM_BOOST_INTEGRATORS_HPP
