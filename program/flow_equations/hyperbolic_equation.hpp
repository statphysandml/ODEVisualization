//
// Created by kades on 8/12/19.
//

#ifndef PROJECT_HYPERBOLIC_EQUATION_HPP
#define PROJECT_HYPERBOLIC_EQUATION_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/flow_equation.hpp"


struct HyperbolicSystemFlowEquation0 : public FlowEquation
{
    HyperbolicSystemFlowEquation0()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return val2 + 2; });
    }
};


struct HyperbolicSystemFlowEquation1 : public FlowEquation
{
    HyperbolicSystemFlowEquation1()
    {}

    void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
    {
        thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return val1 + val2 - 1; });
    }
};


class HyperbolicSystemFlowEquations : public FlowEquationsWrapper
{
public:
    HyperbolicSystemFlowEquations()
    {
        flow_equations = std::vector< FlowEquation* > {
                new HyperbolicSystemFlowEquation0(),
                new HyperbolicSystemFlowEquation1(),
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
        return "hyperbolic_system";
    }

    const static uint8_t dim = 2;

private:
    std::vector < FlowEquation* > flow_equations;
};


#endif //PROJECT_HYPERBOLIC_EQUATION_HPP
