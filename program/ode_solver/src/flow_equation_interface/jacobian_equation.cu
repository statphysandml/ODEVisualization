//
// Created by kades on 7/18/19.
//

// Pre installed equations!!

#include "../../include/flow_equation_interface/jacobian_equation.hpp"
#include "../../../flow_equations/identity_jacobian_equation.hpp"
/* #include "../../../flow_equations/hyperbolic_jacobian.hpp"
#include "../../../flow_equations/3D_hyperbolic_jacobian.hpp" */
// #include "../../flow_equations/scalar_theory_flow_equation.hpp"
#include "../../../flow_equations/three_point_system_jacobian.hpp"
/* #include "../../../flow_equations/four_point_system_jacobian.hpp"
#include "../../../flow_equations/lorentz_attractor_jacobian.hpp"
#include "../../../flow_equations/three_level_system_jacobian.hpp"
#include "../../../flow_equations/small_three_level_system_jacobian.hpp" */

JacobianWrapper *JacobianWrapper::make_jacobian(const std::string theory)
{
    if (theory == "identity")
        return new IdentityJacobianEquations();
        // else if(theory == "scalar_theory")
        //    return ScalarTheoryFlowEquations;
    /* else if(theory == "hyperbolic_system")
        return new HyperbolicSystemJacobianEquations();
    else if(theory == "3D_hyperbolic_system")
        return new ThreeDHyperbolicSystemJacobianEquations(); */
    /* else if(theory == "three_point_system")
        return new ThreePointSystemJacobianEquations(0); */
    /* else if(theory == "four_point_system")
        return new FourPointSystemJacobianEquations(0);
    else if(theory == "lorentz_attractor")
        return new LorentzAttractorJacobianEquations(0);
    else if(theory == "three_level_system")
        return new ThreeLevelSystemJacobianEquations(0);
    else if(theory == "small_three_level_system")
        return new SmallThreeLevelSystemJacobianEquations(0); */
    else
    {
        std::cout << "\nERROR: Flow equation not known" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}