//
// Created by kades on 8/15/19.
//

#ifndef PROJECT_3D_HYPERBOLIC_EQUATION_HPP
#define PROJECT_3D_HYPERBOLIC_EQUATION_HPP


#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/flow_equation.hpp"


struct ThreeDHyperbolicSystemFlowEquation0 : public FlowEquation
{
    ThreeDHyperbolicSystemFlowEquation0()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return val2 + 2; });
    }
};


struct ThreeDHyperbolicSystemFlowEquation1 : public FlowEquation
{
    ThreeDHyperbolicSystemFlowEquation1()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return val1 + val2 - 1; });
    }
};


struct ThreeDHyperbolicSystemFlowEquation2 : public FlowEquation
{
    ThreeDHyperbolicSystemFlowEquation2()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[2].begin(), variables[2].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val3) { return 2.0*val3; });
    }
};


class ThreeDHyperbolicSystemFlowEquations : public FlowEquationsWrapper
{
public:
    ThreeDHyperbolicSystemFlowEquations()
    {
        flow_equations = std::vector< FlowEquation* > {
                new ThreeDHyperbolicSystemFlowEquation0(),
                new ThreeDHyperbolicSystemFlowEquation1(),
                new ThreeDHyperbolicSystemFlowEquation2()
        };
    }

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int dim_index) override
    {
        (*flow_equations[dim_index])(derivatives, variables);
    }

    uint8_t get_dim() override
    {
        return dim;
    }

    static std::string name()
    {
        return "3D_hyperbolic_system";
    }

    const static uint8_t dim = 3;

private:
    std::vector < FlowEquation* > flow_equations;
};

#endif //PROJECT_3D_HYPERBOLIC_EQUATION_HPP
