#include "../include/flow_equations_python.hpp"


namespace odesolver {
    namespace pybind {

        void init_flow_equations(py::module &m)
        {
            auto mflow = m.def_submodule("flow");

            py::class_<flowequations::FlowEquationsWrapper, std::shared_ptr<flowequations::FlowEquationsWrapper>>(mflow, "FlowEquations");

            py::class_<flowequations::JacobianEquationsWrapper, std::shared_ptr<flowequations::JacobianEquationsWrapper>>(mflow, "JacobianEquations");

            mflow.def("compute_flow", [](const devdat::DevDatC &coordinates, std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations) {
                return flowequations::compute_flow(coordinates, flow_equations.get());
            });
            
            mflow.def("compute_jacobian_elements", [](const devdat::DevDatC &coordinates, std::shared_ptr<flowequations::JacobianEquationsWrapper> jacobian_equations) {
                return flowequations::compute_jacobian_elements(coordinates, jacobian_equations.get());
            });
        }
    }
}