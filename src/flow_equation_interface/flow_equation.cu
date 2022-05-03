#include "../../include/flow_equation_interface/flow_equation.hpp"


odesolver::DevDatC compute_vertex_velocities(const odesolver::DevDatC &coordinates, FlowEquationsWrapper * flow_equations)
{
    const uint dim = coordinates.dim_size();
    auto number_of_coordinates = coordinates[0].size();
    odesolver::DevDatC vertex_velocities(dim, number_of_coordinates);
    // Evaluate flow equation for each lambda_{dim_index} separately
    for(auto dim_index = 0; dim_index < dim; dim_index ++) {
        (*flow_equations)(vertex_velocities[dim_index], coordinates, dim_index);
    }
    return std::move(vertex_velocities);
}


void compute_vertex_velocities(const odesolver::DevDatC &coordinates, odesolver::DevDatC &vertex_velocities, FlowEquationsWrapper * flow_equations)
{
    // Evaluate flow equation for each lambda_{dim_index} separately
    for(auto dim_index = 0; dim_index < coordinates.dim_size(); dim_index ++) {
        (*flow_equations)(vertex_velocities[dim_index], coordinates, dim_index);
    }
}

/* #include "../../include/flow_equation_interface/flow_equation.hpp"
#include "../../../flow_equations/identity_flow_equation.hpp" */
// #include "../../flow_equations/scalar_theory_flow_equation.hpp"
/* #include "../../../flow_equations/hyperbolic_equation.hpp"
#include "../../../flow_equations/3D_hyperbolic_equation.hpp" */
// #include "../../../flow_equations/three_point_system_flow_equation.hpp"
/* #include "../../../flow_equations/four_point_system_flow_equation.hpp"
#include "../../../flow_equations/lorentz_attractor_flow_equation.hpp"
#include "../../../flow_equations/three_level_system_flow_equation.hpp"
#include "../../../flow_equations/small_three_level_system_flow_equation.hpp" */

/* FlowEquationsWrapper *FlowEquationsWrapper::make_flow_equation(const std::string theory)
{
    if (theory == "identity")
        return new IdentityFlowEquations();
    // else if(theory == "scalar_theory")
    //    return ScalarTheoryFlowEquations; */
/*    else if(theory == "hyperbolic_system")
        return new HyperbolicSystemFlowEquations();
    else if(theory == "3D_hyperbolic_system")
        return new ThreeDHyperbolicSystemFlowEquations(); */
    /* else if(theory == "three_point_system")
        return new ThreePointSystemFlowEquations(0); */
/*    else if(theory == "four_point_system")
        return new FourPointSystemFlowEquations(0);
    else if(theory == "four_point_system")
        return new LorentzAttractorFlowEquations(0);
    else if(theory == "three_level_system")
        return new ThreeLevelSystemFlowEquations(0);
    else if(theory == "small_three_level_system")
        return new SmallThreeLevelSystemFlowEquations(0); */
    /* else
    {
        std::cout << "\nERROR: Flow equation not known" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}*/