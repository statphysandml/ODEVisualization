//
// Created by lukas on 02.04.19.
//

#ifndef PROJECT_FLOW_EQUATION_HPP
#define PROJECT_FLOW_EQUATION_HPP

#include <vector>
#include <string>

#include "../util/header.hpp"
#include "../util/dev_dat.hpp"

class FlowEquationsWrapper
{
public:
    static FlowEquationsWrapper * make_flow_equation(std::string theory);

    virtual void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int dim_index) = 0;
    virtual uint8_t get_dim() = 0;
    virtual bool pre_installed_theory() = 0;

    static std::string name()
    {
        return "FlowEquationsWrapper";
    }
};

struct FlowEquation
{
    virtual void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) = 0;
};

#endif //PROJECT_FLOW_EQUATION_HPP
