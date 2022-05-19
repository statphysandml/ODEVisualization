#include <odesolver/flow_equations/flow_equation.hpp>


namespace odesolver {
    namespace flowequations {
        json FlowEquationsWrapper::get_json() const {
            return {};
        }

        std::string FlowEquationsWrapper::name()
        {
            return "flow_equation";
        }

        odesolver::DevDatC compute_flow(const odesolver::DevDatC &coordinates, FlowEquationsWrapper * flow_equations)
        {
            const uint dim = coordinates.dim_size();
            auto number_of_coordinates = coordinates.n_elems();
            odesolver::DevDatC vertex_velocities(dim, number_of_coordinates);
            // Evaluate flow equation for each variable_{dim_index} separately
            for(auto dim_index = 0; dim_index < dim; dim_index ++) {
                (*flow_equations)(vertex_velocities[dim_index], coordinates, dim_index);
            }
            return std::move(vertex_velocities);
        }


        void compute_flow(const odesolver::DevDatC &coordinates, odesolver::DevDatC &vertex_velocities, FlowEquationsWrapper * flow_equations)
        {
            // Evaluate flow equation for each variable_{dim_index} separately
            for(auto dim_index = 0; dim_index < coordinates.dim_size(); dim_index ++) {
                (*flow_equations)(vertex_velocities[dim_index], coordinates, dim_index);
            }
        }
    }
}