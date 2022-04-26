//
// Created by lukas on 02.04.19.
//

#ifndef PROJECT_IDENTITYFLOWEQUATION_HPP
#define PROJECT_IDENTITYFLOWEQUATION_HPP

#include "../include/flow_equation_interface/flow_equation.hpp"

#include <cmath>

struct IdentityFlowEquation : public FlowEquation {
    IdentityFlowEquation(const int dim_index_) : dim_index(dim_index_)
    {}

    void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
    {
        // Cuda code
        thrust::transform(variables[dim_index].begin(), variables[dim_index].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &vertex) { return vertex; });
    }

    const int dim_index;
};

class IdentityFlowEquations : public FlowEquationsWrapper
{
public:
    IdentityFlowEquations(const uint8_t dim_=10) : dim(dim_)
    {
        for(int dim_index = 0; dim_index < dim; dim_index++)
            flow_equations.push_back(new IdentityFlowEquation(dim_index));
    }

    void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int dim_index) override
    {
        (*flow_equations[dim_index])(derivatives, variables);
    }

    uint8_t get_dim() override
    {
        return dim;
    }

    bool pre_installed_theory()
    {
        return true;
    }

    static std::string name()
    {
        return "Identity";
    }

private:
    const uint8_t dim;
    std::vector < FlowEquation* > flow_equations;
};

#endif //PROJECT_IDENTITYFLOWEQUATION_HPP
