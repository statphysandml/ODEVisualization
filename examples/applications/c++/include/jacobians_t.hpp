#include <iostream>
#include <vector>
#include <odesolver/header.hpp>
#include <odesolver/modes/jacobians.hpp>


void jacobians_t()
{
    std::vector<std::vector<double>> jacobian_elements {
        {1.0, 0.0, 1.0, -3.0, 2.0, 3.0, 0.0, 2.0, 3.0},
        {1.0, 0.0, 2.0, -3.0, 2.0, 3.0, 0.0, 2.0, 3.0},
        {1.0, 1.0, 1.0, -3.0, 2.0, 3.0, 0.0, 2.0, 3.0},
        {1.0, 4.0, 1.0, -3.0, 2.0, 3.0, 0.0, 2.0, 3.0}
    };

    auto jacobians = odesolver::modes::Jacobians::from_vec_vec(jacobian_elements);
    
    /* std::vector<double> jacobian_elements {
        1.0, 0.0, 1.0, -3.0, 2.0, 3.0, 0.0, 2.0, 3.0,
        1.0, 0.0, 2.0, -3.0, 2.0, 3.0, 0.0, 2.0, 3.0,
        1.0, 1.0, 1.0, -3.0, 2.0, 3.0, 0.0, 2.0, 3.0,
        1.0, 4.0, 1.0, -3.0, 2.0, 3.0, 0.0, 2.0, 3.0
    };

    odesolver::modes::Jacobians jacobians(jacobian_elements, 3); */

    std::cout << jacobians.size() << std::endl;

    jacobians.eval();

    auto eigenvalues = jacobians.get_eigenvalue(1);

    auto jacobians_json = jacobians.jacobians_to_json();

    std::cout << jacobians_json << std::endl;

    auto eigenvectors = jacobians.eigenvectors_to_json();

    std::cout << "Eigenvectors " << eigenvectors << std::endl;
}