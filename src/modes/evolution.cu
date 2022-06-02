#include <odesolver/modes/evolution.hpp>


namespace odesolver {
    namespace modes {
        Evolution::Evolution(
            const json params,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr
        ) : ODEVisualization(params, flow_equations_ptr, jacobians_ptr),
            flow_equations_system_(flow_equations_ptr.get())
        {}

        Evolution Evolution::generate(
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr
        )
        {
            return Evolution(
                {},
                flow_equations_ptr,
                jacobians_ptr
            );
        }

        Evolution Evolution::from_file(
            const std::string rel_config_dir,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr
        )
        {
            return Evolution(
                param_helper::fs::read_parameter_file(
                    param_helper::proj::project_root() + rel_config_dir + "/", "config", false),
                flow_equations_ptr,
                jacobians_ptr
            );
        }
    }
}