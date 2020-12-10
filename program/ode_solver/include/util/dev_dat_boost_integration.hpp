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

namespace boost { namespace numeric { namespace odeint {

            template<>
            struct is_resizeable< DevDatC >
            {
                typedef boost::true_type type;
                static const bool value = type::value;
            };

            template<>
            struct same_size_impl< DevDatC, DevDatC >
            { // define how to check size
                __host__ __device__
                static bool same_size( const DevDatC &v1,
                                       const DevDatC &v2 )
                {
                    return (v1.size() == v2.size()) && (v1.dim_size() == v2.dim_size());
                }
            };

            template<>
            struct resize_impl< DevDatC, DevDatC >
            { // define how to resize
                __host__ __device__
                static void resize( DevDatC &v1,
                                    const DevDatC &v2 )
                {
                    v1.resize( v2.size() );
                    v1.set_dim(v2.dim_size());
                    v1.set_N(v2.size() / v2.dim_size());
                    v1.initialize_dimension_iterators();
                }
            };


        } } }

// Dispatchers

#include <boost/numeric/odeint/algebra/algebra_dispatcher.hpp>

// Specializations for the DevDat
namespace boost {
    namespace numeric {
        namespace odeint {
            template<typename Vec, typename VecIterator, typename ConstVecIterator>
            struct algebra_dispatcher< DevDat<Vec, VecIterator, ConstVecIterator > >
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
            struct operations_dispatcher< DevDat<Vec, VecIterator, ConstVecIterator> >
        {
            typedef thrust_operations operations_type;
        };
    } // namespace odeint
} // namespace numeric
} // namespace boost

#endif //PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP
