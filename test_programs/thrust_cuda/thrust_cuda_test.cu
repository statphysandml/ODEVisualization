#include "helper.hpp"
#include "dev_dat.hpp"

int main(void)
{    
    int driver_version , runtime_version;
    cudaDriverGetVersion( &driver_version );
    cudaRuntimeGetVersion ( &runtime_version );
    std::cout << driver_version << "\t" << runtime_version << std::endl;
    
    // generate 32M random numbers serially
    thrust::host_vector<cudaT> h_vec(2 << 2, 0.1);
    // std::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer data to the device
    dev_vec d_vec = h_vec;
    DevDatC sampled_coordinates(d_vec, 2);

    FlowEquationsWrapper * flow_equations = FlowEquationsWrapper::make_flow_equation("identity", 2);

    /* auto observer = new ConditionalIntersectionObserver(flow_equations);
    Evolution<ConditionalIntersectionObserver> evaluator(flow_equations, observer); */

    flow_equation_system system(flow_equations);
    controlled_stepper rk;
    cudaT t = 0;
    cudaT delta_t = 0.01;
    int observe_every_nth_step = 10;
    auto &initial_coordinates = sampled_coordinates;

    print_range("Initial point", sampled_coordinates.begin(), sampled_coordinates.end());
    boost::numeric::odeint::integrate_n_steps(rk, system, initial_coordinates, t, delta_t, observe_every_nth_step);
    // evaluator.evolve(sampled_coordinates, 0.0, 1.0, 0.04);
    print_range("End point", sampled_coordinates.begin(), sampled_coordinates.end());

    /* print_range("Initial point", d_vec.begin(), d_vec.end());
    boost::numeric::odeint::integrate_n_steps(rk, system, d_vec , t, delta_t, observe_every_nth_step);
    // evaluator.evolve(sampled_coordinates, 0.0, 1.0, 0.04);
    print_range("End point", d_vec.begin(), d_vec.end()); */

    /* auto it_t = sampled_coordinates.begin();

    boost::numeric::odeint::range_algebra rang();
    rang.for_each(it_t , it_t ,
            typename operations_type::template scale_sum2< value_type , time_type >( 1.0 , dt*b21 )*/
    // auto op = boost::numeric::odeint::default_operations::scale_sum2<double, double>();

    /*print_range("Initial point", sampled_coordinates.begin(), sampled_coordinates.end());
    evaluator.evolve_observer_based(sampled_coordinates, delta_t);*/

//     thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), [] __host__ __device__ (const cudaT &vertex) { return vertex; });
    
    return 0;
}