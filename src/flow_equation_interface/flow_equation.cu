//
// Created by kades on 7/18/19.
//

#include "../../include/flow_equation_interface/flow_equation.hpp"
#include "../../../flow_equations/identity_flow_equation.hpp"
// #include "../../flow_equations/scalar_theory_flow_equation.hpp"
/* #include "../../../flow_equations/hyperbolic_equation.hpp"
#include "../../../flow_equations/3D_hyperbolic_equation.hpp" */
#include "../../../flow_equations/three_point_system_flow_equation.hpp"
/* #include "../../../flow_equations/four_point_system_flow_equation.hpp"
#include "../../../flow_equations/lorentz_attractor_flow_equation.hpp"
#include "../../../flow_equations/three_level_system_flow_equation.hpp"
#include "../../../flow_equations/small_three_level_system_flow_equation.hpp" */

FlowEquationsWrapper *FlowEquationsWrapper::make_flow_equation(const std::string theory)
{
    if (theory == "identity")
        return new IdentityFlowEquations();
    // else if(theory == "scalar_theory")
    //    return ScalarTheoryFlowEquations;
/*    else if(theory == "hyperbolic_system")
        return new HyperbolicSystemFlowEquations();
    else if(theory == "3D_hyperbolic_system")
        return new ThreeDHyperbolicSystemFlowEquations(); */
    else if(theory == "three_point_system")
        return new ThreePointSystemFlowEquations(0);
/*    else if(theory == "four_point_system")
        return new FourPointSystemFlowEquations(0);
    else if(theory == "four_point_system")
        return new LorentzAttractorFlowEquations(0);
    else if(theory == "three_level_system")
        return new ThreeLevelSystemFlowEquations(0);
    else if(theory == "small_three_level_system")
        return new SmallThreeLevelSystemFlowEquations(0); */
    else
    {
        std::cout << "\nERROR: Flow equation not known" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}