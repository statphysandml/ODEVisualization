//
// Created by lukas on 15.09.19.
//

#ifndef PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP
#define PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP

#include <boost/numeric/odeint.hpp>

#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include "dev_dat.hpp"

// Dispatchers

#include <boost/numeric/odeint/algebra/algebra_dispatcher.hpp>

// Specializations for the DevDat
namespace boost {
    namespace numeric {
        namespace odeint {
            template<typename Vec, typename VecIterator, typename ConstVecIterator>
            struct algebra_dispatcher< odesolver::DevDat<Vec, VecIterator, ConstVecIterator > >
        {
            typedef thrust_algebra algebra_type;
        };

    } // namespace odeint
} // namespace numeric
} // namespace boost

#include <boost/numeric/odeint/algebra/operations_dispatcher.hpp>

// Support for DevDat
namespace boost {
    namespace numeric {
        namespace odeint {
            template<typename Vec, typename VecIterator, typename ConstVecIterator>
            struct operations_dispatcher< odesolver::DevDat<Vec, VecIterator, ConstVecIterator> >
        {
            typedef thrust_operations operations_type;
        };
    } // namespace odeint
} // namespace numeric
} // namespace boost

#endif //PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP
