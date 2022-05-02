#include "../../include/flow_equation_interface/jacobian_equation.hpp"

odesolver::DevDatC compute_jacobian_elements(const odesolver::DevDatC &coordinates, JacobianEquationWrapper * jacobian_equations)
{
    const uint dim = coordinates.dim_size();
    auto number_of_coordinates = coordinates[0].size();
    odesolver::DevDatC jacobian_elements(pow(dim, 2), number_of_coordinates);
    // Evaluate jacobian for each element separately
    for(auto matrix_idx = 0; matrix_idx < pow(dim, 2); matrix_idx ++) {
        (*jacobian_equations)(jacobian_elements[matrix_idx], coordinates, matrix_idx);
    }
    return std::move(jacobian_elements);
}


/* std::vector<Eigen::MatrixXd*> compute_jacobian_elements(const odesolver::DevDatC &coordinates, JacobianEquationWrapper * jacobian_equations)
{
    const uint dim = coordinates.dim_size();
    auto number_of_coordinates = coordinates[0].size();
    auto jacobians = std::vector<Eigen::MatrixXd*> (number_of_coordinates);

    for(auto coor_idx = 0; coor_idx < number_of_coordinates; coor_idx ++)
        jacobians[coor_idx] = new Eigen::MatrixXd(dim, dim);

    // Evaluate jacobian for each element separately
    for(auto matrix_idx = 0; matrix_idx < pow(dim, 2); matrix_idx ++)
    {
        odesolver::DevDatC jacobian_element_wrapper(1, number_of_coordinates);
        odesolver::DimensionIteratorC &jacobian_element = jacobian_element_wrapper[0];
        // dev_vec jacobian_element(number_of_coordinates);
        (*jacobian_equations)(jacobian_element, coordinates, matrix_idx);
        thrust::host_vector<cudaT> host_jacobian_element(jacobian_element.begin(), jacobian_element.end());
        for(auto coor_idx = 0; coor_idx < number_of_coordinates; coor_idx ++) {
            (*jacobians[coor_idx])(matrix_idx) = host_jacobian_element[coor_idx];
        }
    } */
    /* for(auto coor_idx = 0; coor_idx < number_of_coordinates; coor_idx++) {
        (*jacobians[coor_idx]).transposeInPlace();
    } */
/*    return std::move(jacobians);
} */



/* #include "../../../flow_equations/identity_jacobian_equation.hpp" */
/* #include "../../../flow_equations/hyperbolic_jacobian.hpp"
#include "../../../flow_equations/3D_hyperbolic_jacobian.hpp" */
// #include "../../flow_equations/scalar_theory_flow_equation.hpp"
// #include "../../../flow_equations/three_point_system_jacobian.hpp"
/* #include "../../../flow_equations/four_point_system_jacobian.hpp"
#include "../../../flow_equations/lorentz_attractor_jacobian.hpp"
#include "../../../flow_equations/three_level_system_jacobian.hpp"
#include "../../../flow_equations/small_three_level_system_jacobian.hpp" */

/* JacobianEquationWrapper *JacobianEquationWrapper::make_jacobian(const std::string theory)
{
    if (theory == "identity")
        return new IdentityJacobianEquations(); */
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
    /* else
    {
        std::cout << "\nERROR: Flow equation not known" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}*/
