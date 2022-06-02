#include <odesolver/boost/dev_dat_boost_integration.hpp>

namespace boost { namespace numeric { namespace odeint {

    bool same_size_impl<odesolver::DevDatC, odesolver::DevDatC>::same_size(const odesolver::DevDatC &v1,
                               const odesolver::DevDatC &v2)
    {
        return (v1.size() == v2.size()) && (v1.dim_size() == v2.dim_size()) && (v1.n_elems() == v2.n_elems());
    }

    void resize_impl<odesolver::DevDatC, odesolver::DevDatC>::resize(odesolver::DevDatC &v1,
                            const odesolver::DevDatC &v2)
    {
        v1.resize(v2.dim_size(), v2.n_elems());
    }

} } }