#include "../../include/odesolver/util/dev_dat_boost_integration.hpp"

namespace boost { namespace numeric { namespace odeint {

    template<>
    struct is_resizeable< odesolver::DevDatC >
    {
        typedef boost::true_type type;
        static const bool value = type::value;
    };

    template<>
    struct same_size_impl< odesolver::DevDatC, odesolver::DevDatC >
    { // define how to check size
        __host__ __device__
        static bool same_size( const odesolver::DevDatC &v1,
                               const odesolver::DevDatC &v2 );
    };

    template<>
    struct resize_impl< odesolver::DevDatC, odesolver::DevDatC >
    { // define how to resize
        __host__ __device__
        static void resize( odesolver::DevDatC &v1,
                            const odesolver::DevDatC &v2 );
    };

} } }