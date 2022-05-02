#ifndef PROGRAM_ODEVISUALISATION_PARAMETERS_HPP
#define PROGRAM_ODEVISUALISATION_PARAMETERS_HPP

#include <iostream>

#include <param_helper/params.hpp>
#include <param_helper/filesystem.hpp>

// #include "path_parameters.hpp"
#include "computation_parameters.hpp"
#include "../../flow_equation_interface/flow_equation.hpp"
#include "../../flow_equation_interface/jacobian_equation.hpp"


using json = nlohmann::json;


class ODEVisualisation : public param_helper::params::Parameters {
public:

    explicit ODEVisualisation(
        const json params,
        std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr=nullptr,
        std::shared_ptr<JacobianEquationWrapper> jacobians_ptr=nullptr,
        const std::string computation_parameters_path=param_helper::proj::project_root()
    ) : Parameters(params),
        flow_equations_ptr_(flow_equations_ptr),
        jacobians_ptr_(jacobians_ptr),
        computation_parameters_path_(computation_parameters_path)
    {
        // Merge specific information and global_parameters.json file for a possible consecutive computation including all parameters
        if(flow_equations_ptr_)
        {
            append_parameters(*flow_equations_ptr_.get());
        }
        
        if(param_helper::fs::check_if_parameter_file_exists(computation_parameters_path_, "computation_parameters", false))
        {
            std::cout << "Computations parameters are loaded from computation_parameters.json" << std::endl;
            json computation_params = param_helper::fs::read_parameter_file(computation_parameters_path_, "computation_parameters", false);
            computation_parameters_ = ComputationParameters(computation_params);
        }
        else
        {
            std::cout << "Computations are performed with default computation parameters - change them by providing a computation_parameters_path or with the member function: set_computation_parameters(), or in the automatically generated computation_parameters.json file." << std::endl;
        }
    }

    void write_configs_to_file(const std::string& rel_config_dir) {
        this->write_to_file(param_helper::proj::project_root() + rel_config_dir + "/", "config", false);
    }

    void set_flow_equations(std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr)
    {
        flow_equations_ptr_ = flow_equations_ptr;
    }

    FlowEquationsWrapper* get_flow_equations()
    {
        return flow_equations_ptr_.get();
    }

    void set_jacobians(std::shared_ptr<JacobianEquationWrapper> jacobians_ptr)
    {
        jacobians_ptr_ = jacobians_ptr;
    }

    JacobianEquationWrapper* get_jacobians()
    {
        return jacobians_ptr_.get();
    }

    // Keep, or remove?
    void set_computation_parameters(
        const int number_of_cubes_per_gpu_call = 20000,
        const int maximum_number_of_gpu_calls = 1000
    )
    {
        computation_parameters_ = ComputationParameters(number_of_cubes_per_gpu_call, maximum_number_of_gpu_calls);
    }

protected:
    std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr_;
    std::shared_ptr<JacobianEquationWrapper> jacobians_ptr_;
    const std::string computation_parameters_path_;

    ComputationParameters computation_parameters_;


};

#endif //PROGRAM_ODEVISUALISATION_PARAMETERS_HPP
