#ifndef PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP
#define PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>
// Dispatchers
#include <boost/numeric/odeint/algebra/algebra_dispatcher.hpp>

#include <odesolver/dev_dat.hpp>

namespace boost { namespace numeric { namespace odeint {

    template<>
    struct is_resizeable<odesolver::DevDatC>
    {
        typedef boost::true_type type;
        static const bool value = type::value;
    };

    template<>
    struct same_size_impl<odesolver::DevDatC, odesolver::DevDatC>
    { // define how to check size
        static bool same_size(const odesolver::DevDatC &v1,
                              const odesolver::DevDatC &v2);
    };

    template<>
    struct resize_impl<odesolver::DevDatC, odesolver::DevDatC>
    { // define how to resize
        static void resize(odesolver::DevDatC &v1,
                           const odesolver::DevDatC &v2);
    };

} } }


// Specializations for DevDat
namespace boost {
    namespace numeric {
        namespace odeint {
            template<typename Vec, typename VecIterator, typename ConstVecIterator>
            struct algebra_dispatcher<odesolver::DevDat<Vec, VecIterator, ConstVecIterator>>
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
            struct operations_dispatcher< odesolver::DevDat<Vec, VecIterator, ConstVecIterator>>
        {
            typedef thrust_operations operations_type;
        };
    } // namespace odeint
} // namespace numeric
} // namespace boost

#endif //PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP
