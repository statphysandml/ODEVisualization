#include "helper.hpp"

FlowEquationsWrapper *FlowEquationsWrapper::make_flow_equation(const std::string flow_equation_name, const uint8_t dim)
{
    return new IdentityFlowEquations(dim);
}
