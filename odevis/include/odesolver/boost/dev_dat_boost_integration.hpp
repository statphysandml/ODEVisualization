#ifndef PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP
#define PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP

#include <boost/numeric/odeint.hpp>
#include <devdat/devdat.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_algebra.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_operations.hpp>

namespace boost { namespace numeric { namespace odeint {

    template<>
    struct is_resizeable<devdat::DevDatC>
    {
        typedef boost::true_type type;
        static const bool value = type::value;
    };

    template<>
    struct same_size_impl<devdat::DevDatC, devdat::DevDatC>
    {
        static bool same_size(const devdat::DevDatC &v1,
                              const devdat::DevDatC &v2);
    };

    template<>
    struct resize_impl<devdat::DevDatC, devdat::DevDatC>
    {
        static void resize(devdat::DevDatC &v1,
                           const devdat::DevDatC &v2);
    };

    template<typename Vec, typename VecIterator, typename ConstVecIterator>
    struct algebra_dispatcher<devdat::DevDat<Vec, VecIterator, ConstVecIterator>>
    {
        typedef thrust_algebra algebra_type;
    };

    template<typename Vec, typename VecIterator, typename ConstVecIterator>
    struct operations_dispatcher<devdat::DevDat<Vec, VecIterator, ConstVecIterator>>
    {
        typedef thrust_operations operations_type;
    };

} } }

#endif //PROGRAM_DEV_DAT_BOOST_INTEGRATION_HPP
