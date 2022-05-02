//
// Created by lukas on 02.04.19.
//

#ifndef PROJECT_FLOW_EQUATION_HPP
#define PROJECT_FLOW_EQUATION_HPP

#include <vector>
#include <string>

#include <param_helper/json.hpp>

#include "../odesolver/util/header.hpp"
#include "../odesolver/util/dev_dat.hpp"

using json = nlohmann::json;

class FlowEquationsWrapper
{
public:
    virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int dim_index) = 0;
    
    virtual size_t get_dim() = 0;

    virtual json get_json() const {
        return {};
    }

    static std::string name()
    {
        return "flow_equation";
    }
};

struct FlowEquation
{
    virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) = 0;
};

template<typename FlowEquations, typename... Args>
std::shared_ptr<FlowEquationsWrapper> generate_flow_equations(Args... args)
{
    return std::make_shared<FlowEquations>(args...);
}

odesolver::DevDatC compute_vertex_velocities(const odesolver::DevDatC &coordinates, FlowEquationsWrapper * flow_equations);

#endif //PROJECT_FLOW_EQUATION_HPP
