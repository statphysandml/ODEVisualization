//
// Created by kades on 5/21/19.
//

#ifndef PROJECT_IDENTITY_JACOBIAN_EQUATION_HPP
#define PROJECT_IDENTITY_JACOBIAN_EQUATION_HPP

#include "../include/flow_equation_interface/jacobian_equation.hpp"


struct IdentityJacobianEquation : public JacobianEquation {
    IdentityJacobianEquation()
    {}

    void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override
    {
        // Cuda code
        thrust::transform(variables[0].begin(), variables[0].end(), derivatives.begin(), [] __host__ __device__ (const float &elem) { return 1.0; });
    }
};

class IdentityJacobianEquations : public JacobianWrapper
{
public:
    IdentityJacobianEquations(const uint8_t dim_=10) : dim(dim_)
    {
        for(int matrix_idx = 0; matrix_idx < pow(dim, 2); matrix_idx++)
            jacobian_equations.push_back(new IdentityJacobianEquation());
    }

    void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int row_idx, const int col_idx) override
    {
        (*jacobian_equations[row_idx * dim + col_idx])(derivatives, variables);
    }

    void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int matrix_idx) override
    {
        (*jacobian_equations[matrix_idx])(derivatives, variables);
    }

    uint8_t get_dim() override
    {
        return dim;
    }

private:
    const uint8_t dim;
    std::vector < JacobianEquation* > jacobian_equations;
};

#endif //PROJECT_IDENTITY_JACOBIAN_EQUATION_HPP
