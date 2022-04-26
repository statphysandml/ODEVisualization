//
// Created by lukas on 25.07.19.
//

#ifndef PROGRAM_FRGVISUALIZATION_PARAMETERS_HPP
#define PROGRAM_FRGVISUALIZATION_PARAMETERS_HPP

#include <param_helper/params.hpp>
#include <param_helper/filesystem.hpp>

#include "path_parameters.hpp"
#include "computation_parameters.hpp"

#include <iostream>

using json = nlohmann::json;


class FRGVisualizationParameters : public param_helper::params::Parameters {
public:

    explicit FRGVisualizationParameters(const json params, const PathParameters path_parameters) : Parameters(params),
        mode_(get_entry<std::string>("mode")),
        path_parameters_(path_parameters),
        // ToDo: Needs to be adapted to work for both scenarios
        // global_params(read_parameter_file(gcp() + "/flow_equations/source/" + path_parameters.theory + "/", "global_parameters", false))
        global_params_(param_helper::fs::read_parameter_file(param_helper::proj::project_root() + "/flow_equations/" + path_parameters.theory_ + "/", "global_parameters", false))
    {
        // Merge specific information and global_parameters.json file for a possible consecutive computation including all parameters
        params_ = param_helper::params::merge(params_, global_params_);
        // Load computation parameters
        load_computation_parameters_from_file();
    }

    void set_computation_parameters(const int number_of_cubes_per_gpu_call = 20000,
                                           const int maximum_number_of_gpu_calls = 1000)
    {
        computation_parameters_ = ComputationParameters(number_of_cubes_per_gpu_call, maximum_number_of_gpu_calls);
        if(writing_to_file_)
            computation_parameters_.write_to_file(path_parameters_.get_base_path(), "computation_parameters", path_parameters_.relative_path_);
    }

    void load_computation_parameters_from_file()
    {
        if(param_helper::fs::check_if_parameter_file_exists(path_parameters_.get_base_path(), "computation_parameters", path_parameters_.relative_path_))
        {
            json computation_params = param_helper::fs::read_parameter_file(path_parameters_.get_base_path(), "computation_parameters", path_parameters_.relative_path_);
            computation_parameters_ = ComputationParameters(computation_params);
        }
        else
        {
            std::cout << "Computations are performed with default computation parameters - change them with the function: set_computation_parameters for your main module parameters" << std::endl;
        }
    }

    void write_to_file(const std::string& results_dir) {
        json output_json = prepare_output_json(results_dir);
        param_helper::fs::write_parameter_file(output_json, path_parameters_.get_base_path() + results_dir + "/", "config", path_parameters_.relative_path_);
    }

    std::string get_mode() const
    {
        return mode_;
    }

    std::string get_theory() const
    {
        return path_parameters_.theory_;
    }

    std::string get_root_dir() const
    {
        return path_parameters_.root_dir_;
    }

    bool get_relative_path () const
    {
        return path_parameters_.relative_path_;
    }

    PathParameters get_path_parameters() const
    {
        return path_parameters_;
    }

protected:
    const std::string mode_;
    PathParameters path_parameters_;
    const json global_params_;
    bool writing_to_file_ = false;

    ComputationParameters computation_parameters_;

    json prepare_output_json(const std::string results_dir)
    {
        writing_to_file_ = true;

        std::string path = param_helper::proj::get_path_to(path_parameters_.get_base_path(), path_parameters_.relative_path_);
        // ToDo: Needs to be adapted to work for both scenarios
        // json global_parameters = read_parameter_file(gcp() + "/flow_equations/source/" + path_parameters.theory + "/", "global_parameters", false);
        json global_parameters = param_helper::fs::read_parameter_file(param_helper::proj::project_root() + "/flow_equations/" + path_parameters_.theory_ + "/", "global_parameters", false);
        param_helper::fs::generate_directory_if_not_present(path_parameters_.get_base_path()); // Base path
        param_helper::fs::generate_directory_if_not_present(path_parameters_.get_base_path() + results_dir + "/"); // Results dir

        // Override computation parameter file
        computation_parameters_.write_to_file(path_parameters_.get_base_path(), "computation_parameters", path_parameters_.relative_path_);

        return param_helper::params::subtract(params_, global_parameters);
    }
};

#endif //PROGRAM_FRGVISUALIZATION_PARAMETERS_HPP
