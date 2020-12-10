//
// Created by lukas on 15.09.19.
//

#ifndef PROGRAM_FLOW_EQUATION_SYSTEM_HPP
#define PROGRAM_FLOW_EQUATION_SYSTEM_HPP

#include "../flow_equation_interface/flow_equation.hpp"


/* Struct for the evolution with boost */
struct FlowEquationSystem
{
    FlowEquationSystem(FlowEquationsWrapper * flow_equations_) : flow_equations(flow_equations_), dim(flow_equations_->get_dim())
    {}

    // https://www.boost.org/doc/libs/1_70_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/state_types__algebras_and_operations.html
    template< class State, class Deriv >
    void operator()( const State &x, Deriv &dxdt, cudaT t ) const {
        for (auto dim_index = 0; dim_index < dim; dim_index++) {
            (*flow_equations)(dxdt[dim_index], x, dim_index);
        }
        /* print_range("State", x.begin(), x.end());
        print_range("Deriv", dxdt.begin(), dxdt.end()); */
    }

    uint8_t get_dim() const {
        return dim;
    }

    FlowEquationsWrapper * flow_equations;
    uint8_t dim;
};

/* Operations that can be applied on both, vertices as well as regular coordinates */
DevDatC compute_vertex_velocities(const DevDatC &coordinates, FlowEquationsWrapper * flow_equations);

#endif //PROGRAM_FLOW_EQUATION_SYSTEM_HPP
