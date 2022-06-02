#include "../include/flow_equations_python.hpp"


namespace odesolver {
    namespace pybind {

        void init_flow_equations(py::module &m)
        {
            auto mflow = m.def_submodule("flow");

            py::class_<odesolver::flowequations::FlowEquationsWrapper, std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper>>(mflow, "FlowEquations");

            py::class_<odesolver::flowequations::JacobianEquationsWrapper, std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper>>(mflow, "JacobianEquations");

            mflow.def("compute_flow", [](const odesolver::DevDatC &coordinates, std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations) {
                return odesolver::flowequations::compute_flow(coordinates, flow_equations.get());
            });
            
            mflow.def("compute_jacobian_elements", [](const odesolver::DevDatC &coordinates, std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobian_equations) {
                return odesolver::flowequations::compute_jacobian_elements(coordinates, jacobian_equations.get());
            });
        }
    }
}