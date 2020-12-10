//
// Created by kades on 8/12/19.
//

#ifndef PROJECT_HYPERBOLIC_JACOBIAN_HPP
#define PROJECT_HYPERBOLIC_JACOBIAN_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/jacobian_equation.hpp"


struct HyperbolicSystemJacobianEquation0 : public JacobianEquation
{
    HyperbolicSystemJacobianEquation0()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 0; });
    }
};


struct HyperbolicSystemJacobianEquation1 : public JacobianEquation
{
    HyperbolicSystemJacobianEquation1()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 1; });
    }
};


struct HyperbolicSystemJacobianEquation2 : public JacobianEquation
{
    HyperbolicSystemJacobianEquation2()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 1; });
    }
};


struct HyperbolicSystemJacobianEquation3 : public JacobianEquation
{
    HyperbolicSystemJacobianEquation3()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 1; });
    }
};


class HyperbolicSystemJacobianEquations : public JacobianWrapper
{
public:
    HyperbolicSystemJacobianEquations()
    {
        jacobian_equations = std::vector< JacobianEquation* > {
                new HyperbolicSystemJacobianEquation0(),
                new HyperbolicSystemJacobianEquation1(),
                new HyperbolicSystemJacobianEquation2(),
                new HyperbolicSystemJacobianEquation3()
        };
    }

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int row_idx, const int col_idx) override
    {
        (*jacobian_equations[row_idx * dim + col_idx])(derivatives, variables);
    }

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int matrix_idx) override
    {
        (*jacobian_equations[matrix_idx])(derivatives, variables);
    }

    uint8_t get_dim() override
    {
        return dim;
    }

    static std::string name()
    {
        return "hyperbolic_system";
    }

    const static uint8_t dim = 2;

private:
    std::vector < JacobianEquation* > jacobian_equations;
};


#endif //PROJECT_HYPERBOLIC_JACOBIAN_HPP
