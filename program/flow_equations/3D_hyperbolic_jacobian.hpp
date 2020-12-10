//
// Created by kades on 8/15/19.
//

#ifndef PROJECT_3D_HYPERBOLIC_JACOBIAN_HPP
#define PROJECT_3D_HYPERBOLIC_JACOBIAN_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/jacobian_equation.hpp"


struct ThreeDHyperbolicSystemJacobianEquation0 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation0()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 0; });
    }
};


struct ThreeDHyperbolicSystemJacobianEquation1 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation1()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 1; });
    }
};


struct ThreeDHyperbolicSystemJacobianEquation2 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation2()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 0; });
    }
};


struct ThreeDHyperbolicSystemJacobianEquation3 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation3()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 1; });
    }
};


struct ThreeDHyperbolicSystemJacobianEquation4 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation4()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 1; });
    }
};


struct ThreeDHyperbolicSystemJacobianEquation5 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation5()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 0; });
    }
};


struct ThreeDHyperbolicSystemJacobianEquation6 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation6()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 0; });
    }
};


struct ThreeDHyperbolicSystemJacobianEquation7 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation7()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 0; });
    }
};


struct ThreeDHyperbolicSystemJacobianEquation8 : public JacobianEquation
{
    ThreeDHyperbolicSystemJacobianEquation8()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 2.0; });
    }
};

class ThreeDHyperbolicSystemJacobianEquations : public JacobianWrapper
{
public:
    ThreeDHyperbolicSystemJacobianEquations()
    {
        jacobian_equations = std::vector< JacobianEquation* > {
                new ThreeDHyperbolicSystemJacobianEquation0(),
                new ThreeDHyperbolicSystemJacobianEquation1(),
                new ThreeDHyperbolicSystemJacobianEquation2(),
                new ThreeDHyperbolicSystemJacobianEquation3(),
                new ThreeDHyperbolicSystemJacobianEquation4(),
                new ThreeDHyperbolicSystemJacobianEquation5(),
                new ThreeDHyperbolicSystemJacobianEquation6(),
                new ThreeDHyperbolicSystemJacobianEquation7(),
                new ThreeDHyperbolicSystemJacobianEquation8()
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
        return "3D_hyperbolic_system";
    }

    const static uint8_t dim = 2;

private:
    std::vector < JacobianEquation* > jacobian_equations;
};



#endif //PROJECT_3D_HYPERBOLIC_JACOBIAN_HPP
