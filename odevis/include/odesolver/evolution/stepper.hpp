#ifndef PROGRAM_STEPPER_HPP
#define PROGRAM_STEPPER_HPP

#include <param_helper/params.hpp>

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <odesolver/boost/dev_dat_boost_integration.hpp>
#include <boost/numeric/odeint/stepper/generation/make_dense_output.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>


namespace odesolver {
    namespace evolution {
        namespace stepper {

            struct RungaKutta4
            {
                typedef boost::numeric::odeint::runge_kutta4<devdat::DevDatC, cudaT, devdat::DevDatC, cudaT> stepper_type;

                RungaKutta4();

                static const std::string concept()
                {
                    return "stepper";
                }

                stepper_type stepper_;
            };

            struct SymplecticRKNSB3McLachlan
            {
                typedef boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan<devdat::DevDatC, devdat::DevDatC, cudaT, devdat::DevDatC, devdat::DevDatC, cudaT> stepper_type;

                SymplecticRKNSB3McLachlan();

                static const std::string concept()
                {
                    return "stepper";
                }

                stepper_type stepper_;
            };

            struct RungaKuttaDopri5
            {
                typedef boost::numeric::odeint::runge_kutta_dopri5<devdat::DevDatC, cudaT, devdat::DevDatC, cudaT> stepper_type; // stepper

                RungaKuttaDopri5();

                static const std::string concept()
                {
                    return "error_stepper";
                }

                stepper_type stepper_;
            };

            template<typename Stepper>
            struct ControlledRungaKutta
            {
                typedef typename boost::numeric::odeint::result_of::make_dense_output< typename Stepper::stepper_type >::type dense_stepper_type;
                
                ControlledRungaKutta(double abs_err_tolerance=1.0e-06, double rel_err_tolerance=1.0e-6) : stepper_(boost::numeric::odeint::make_dense_output(abs_err_tolerance, rel_err_tolerance, Stepper().stepper_))
                {}

                static const std::string concept()
                {
                    return "dense_output_stepper";
                }

                dense_stepper_type stepper_;
            };
        }
    }
}

#endif //PROGRAM_EVOLUTION_HPP
