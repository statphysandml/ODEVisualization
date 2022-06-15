#ifndef PROGRAM_ODEVISUALIZATION_PARAMETERS_HPP
#define PROGRAM_ODEVISUALIZATION_PARAMETERS_HPP

#include <iostream>

#include <param_helper/params.hpp>
#include <param_helper/filesystem.hpp>

#include <odesolver/flow_equations/flow_equation.hpp>
#include <odesolver/flow_equations/jacobian_equation.hpp>


using json = nlohmann::json;


namespace odesolver {
    namespace modes {
        class ODEVisualization : public param_helper::params::Parameters {
        public:

            explicit ODEVisualization(
                const json params,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr=nullptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr=nullptr
            ) : Parameters(params),
                flow_equations_ptr_(flow_equations_ptr),
                jacobians_ptr_(jacobians_ptr)
            {
                // Merge specific information and global_parameters.json file for a possible consecutive computation including all parameters
                if(flow_equations_ptr_)
                {
                    append_parameters(*flow_equations_ptr_.get());
                }
            }

            void write_configs_to_file(const std::string& rel_config_dir) {
                this->write_to_file(param_helper::proj::project_root() + rel_config_dir + "/", "config", false);
            }

            void set_flow_equations(std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr)
            {
                flow_equations_ptr_ = flow_equations_ptr;
            }

            odesolver::flowequations::FlowEquationsWrapper* get_flow_equations()
            {
                return flow_equations_ptr_.get();
            }

            void set_jacobians(std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr)
            {
                jacobians_ptr_ = jacobians_ptr;
            }

            odesolver::flowequations::JacobianEquationsWrapper* get_jacobians()
            {
                return jacobians_ptr_.get();
            }

        protected:
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr_;
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr_;
        };
    }
}

#endif //PROGRAM_ODEVISUALIZATION_PARAMETERS_HPP
