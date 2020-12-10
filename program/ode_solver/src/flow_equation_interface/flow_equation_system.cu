#include "../../include/flow_equation_interface/flow_equation_system.hpp"

DevDatC compute_vertex_velocities(const DevDatC &coordinates, FlowEquationsWrapper * flow_equations)
{
    const uint dim = flow_equations->get_dim();
    auto number_of_coordinates = coordinates[0].size();
    DevDatC vertex_velocities(dim, number_of_coordinates);
    // Evaluate flow equation for each lambda_{dim_index} separately
    for(auto dim_index = 0; dim_index < dim; dim_index ++) {
        (*flow_equations)(vertex_velocities[dim_index], coordinates, dim_index);
    }
    return vertex_velocities;
}