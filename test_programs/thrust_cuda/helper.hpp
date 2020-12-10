//
// Created by lukas on 14.09.19.
//

#ifndef THRUST_CUDA_HELPER_HPP
#define THRUST_CUDA_HELPER_HPP

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#include "header.hpp"
#include "dev_dat.hpp"

//[ Print range

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
    std::cout << "\n";
}

//]

//[ Flow equations

struct FlowEquation
{
    virtual void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) = 0;
};

struct IdentityFlowEquation : FlowEquation {
    IdentityFlowEquation(const int dim_index_) : dim_index(dim_index_)
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[dim_index].begin(), variables[dim_index].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &vertex) { return vertex; });
    }
    const int dim_index;
};

class FlowEquationsWrapper
{
public:
    static FlowEquationsWrapper * make_flow_equation(std::string flow_equation_name, uint8_t dim);

    virtual void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int dim_index) = 0;
};

class IdentityFlowEquations : public FlowEquationsWrapper
{
public:
    IdentityFlowEquations(const uint8_t dim_) : dim(dim_)
    {
        for(int dim_index = 0; dim_index < dim; dim_index++)
            flow_equations.push_back(new IdentityFlowEquation(dim_index));
    }

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int dim_index) override
    {
        (*flow_equations[dim_index])(derivatives, variables);
    }
private:
    const uint8_t dim;
    std::vector < FlowEquation* > flow_equations;
};

//]

//[ Observer
struct EvolutionObserver
{
    virtual void operator() (const DevDatC &coordinates, cudaT t) = 0;
};

struct ConditionalRangeObserver : public EvolutionObserver
{
    void operator() (const DevDatC &coordinates, cudaT t) override
    {}
};

struct ConditionalIntersectionObserver : public ConditionalRangeObserver {
    ConditionalIntersectionObserver(FlowEquationsWrapper * flow_equations_) : flow_equations(flow_equations_)
    {}

    void operator() (const DevDatC &coordinates, cudaT t) override
    {}

    FlowEquationsWrapper * flow_equations;
};

//]

//[ Evolution

#include <boost/numeric/odeint.hpp>

#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

namespace boost { namespace numeric { namespace odeint {

            template<>
            struct is_resizeable< DevDatC >
            {
                typedef boost::true_type type;
                static const bool value = type::value;
            };

            template< >
            struct same_size_impl< DevDatC, DevDatC >
            { // define how to check size
                __host__ __device__
                static bool same_size( const DevDatC &v1,
                                       const DevDatC &v2 )
                {
                    return (v1.size() == v2.size()) && (v1.dim_size() == v2.dim_size());
                }
            };

            template< >
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


// #include <boost/numeric/odeint/external/thrust/thrust_algebra.hpp>
#include <boost/numeric/odeint/algebra/algebra_dispatcher.hpp>

// specializations for the standard thrust containers

namespace boost {
    namespace numeric {
        namespace odeint {

// specialization for thrust host_vector
            /* template< class T , class A >
            struct algebra_dispatcher< thrust::host_vector< T , A > > */
            template<typename Vec, typename VecIterator, typename ConstVecIterator>
            struct algebra_dispatcher< DevDat<Vec, VecIterator, ConstVecIterator > >
        {
            typedef thrust_algebra algebra_type;
        };

// specialization for thrust device_vector
/*        template< class T , class A >
        struct algebra_dispatcher< thrust::device_vector< T , A > >
    {
        typedef thrust_algebra algebra_type;
    };*/

} // namespace odeint
} // namespace numeric
} // namespace boost

#include <boost/numeric/odeint/external/thrust/thrust_operations.hpp>
#include <boost/numeric/odeint/algebra/operations_dispatcher.hpp>

// support for the standard thrust containers

namespace boost {
    namespace numeric {
        namespace odeint {

// specialization for thrust host_vector
            template<typename Vec, typename VecIterator, typename ConstVecIterator>
            struct operations_dispatcher< DevDat<Vec, VecIterator, ConstVecIterator> >
        {
            typedef thrust_operations operations_type;
        };

/*// specialization for thrust device_vector
        template< class T , class A >
        struct operations_dispatcher< thrust::device_vector< T , A > >
    {
        typedef thrust_operations operations_type;
    };*/

} // namespace odeint
} // namespace numeric
} // namespace boost


typedef boost::numeric::odeint::runge_kutta_dopri5< DevDatC, cudaT,  DevDatC, cudaT > controlled_stepper;
// typedef boost::numeric::odeint::runge_kutta_dopri5< dev_vec , cudaT,  dev_vec , cudaT > controlled_stepper;
typedef boost::numeric::odeint::runge_kutta4< DevDatC, cudaT,  DevDatC, cudaT > constant_stepper;

struct flow_equation_system
{
    flow_equation_system(FlowEquationsWrapper * flow_equations_) : flow_equations(flow_equations_)
    {}

    // https://www.boost.org/doc/libs/1_70_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/state_types__algebras_and_operations.html
    template< class State, class Deriv >
    void operator()( const State &x, Deriv &dxdt, cudaT t ) const {
        for (auto dim_index = 0; dim_index < 2; dim_index++) {
            (*flow_equations)(dxdt[dim_index], x, dim_index);
        }
        /* print_range("State", x.begin(), x.end());
        print_range("Deriv", dxdt.begin(), dxdt.end()); */
    }

    FlowEquationsWrapper * flow_equations;
};

template <typename stepper_type, typename Observer>
struct Integrator
{
    virtual void operator() (stepper_type &stepper, flow_equation_system &system, DevDatC &initial_coordinates, cudaT &start_t, cudaT &end_t, cudaT &delta_t, Observer * observer = nullptr) = 0;
};

template <typename stepper_type, typename Observer>
struct AdaptiveIntegrator : public Integrator<stepper_type, Observer>
{
    void operator() (stepper_type &stepper, flow_equation_system &system, DevDatC &initial_coordinates, cudaT &start_t, cudaT &end_t, cudaT &delta_t, Observer * observer = nullptr) override
    {
        /* if(observer)
            boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled(1.0e-6, 1.0e-6, stepper), system, initial_coordinates, start_t, end_t, delta_t, *observer);
        else
            boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled(1.0e-6, 1.0e-6, stepper), system, initial_coordinates, start_t, end_t, delta_t);*/
    }
};

template<typename Observer>
class Evolution
{
public:
    Evolution(FlowEquationsWrapper * flow_equations_, Observer * observer_ = nullptr) :
            system(flow_equation_system(flow_equations_)), observer(observer_)
    {}

    void evolve(DevDatC &initial_coordinates, const cudaT start_t=0.0, const cudaT end_t=1.0, const cudaT delta_t=0.01)
    {
        std::cout << "Performing adaptive integration" << std::endl;
        auto integrator = new AdaptiveIntegrator<controlled_stepper, Observer> ();
        controlled_stepper rk;
        evolve_< controlled_stepper > (rk, integrator, initial_coordinates, start_t, end_t, delta_t);
        delete integrator;
    }
private:
    // ToDo: The dim variable can be removed at many points since present in flow equations, etc. (see also in other files
    flow_equation_system system;
    Observer * observer;

    template <typename stepper_type >
    void evolve_(stepper_type stepper, Integrator<stepper_type, Observer> * integrator, DevDatC &initial_coordinates, cudaT start_t=0.0, cudaT end_t=1.0, cudaT delta_t=0.01)
    {
        int observe_every_nth_step = 10;
        int maximum_total_number_of_steps = 100;

        if(observer and observe_every_nth_step > 1)
        {
            // Note that the integrator is not used in this case
            cudaT t = start_t;
            uint number_of_steps = 0;
            while(t < end_t && number_of_steps <= maximum_total_number_of_steps)
            {
                // ToDo: Check if this is possible with contralled steppers, i.e. to have a variable delta t
                // boost::numeric::odeint::integrate_n_steps(stepper, system, initial_coordinates, t, delta_t, observe_every_nth_step);
                (*observer)(initial_coordinates, t);
                t += observe_every_nth_step * delta_t;
                number_of_steps += observe_every_nth_step;
            }
        }
        else
            (*integrator) (stepper, system, initial_coordinates, start_t, end_t, delta_t, observer);
    }
};

//]

#endif //THRUST_CUDA_HELPER_HPP
