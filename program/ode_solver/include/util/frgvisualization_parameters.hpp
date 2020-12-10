//
// Created by lukas on 25.07.19.
//

#ifndef PROGRAM_FRGVISUALIZATION_PARAMETERS_HPP
#define PROGRAM_FRGVISUALIZATION_PARAMETERS_HPP

#include "param_helper/params.hpp"
#include "path_parameters.hpp"
#include "computation_parameters.hpp"

#include <iostream>

using json = nlohmann::json;


class FRGVisualizationParameters : public Parameters {
public:

    explicit FRGVisualizationParameters(const json params_, const PathParameters path_parameters_) : Parameters(params_),
        mode(get_value_by_key<std::string>("mode")),
        path_parameters(path_parameters_),
        // ToDo: Needs to be adapted to work for both scenarios
        // global_params(read_parameter_file(gcp() + "/flow_equations/source/" + path_parameters.theory + "/", "global_parameters", false))
        global_params(read_parameter_file(gcp() + "/flow_equations/" + path_parameters.theory + "/", "global_parameters", false))
    {
        // Merge specific information and global_parameters.json file for a possible consecutive computation including all parameters
        params = merge(params, global_params);
        // Load computation parameters
        load_computation_parameters_from_file();
    }

    void set_computation_parameters(const int number_of_cubes_per_gpu_call = 20000,
                                           const int maximum_number_of_gpu_calls = 1000)
    {
        computation_parameters = ComputationParameters(number_of_cubes_per_gpu_call, maximum_number_of_gpu_calls);
        if(writing_to_file)
            computation_parameters.write_to_file(path_parameters.get_base_path(), "computation_parameters", path_parameters.relative_path);
    }

    void load_computation_parameters_from_file()
    {
        if(check_if_parameter_file_exists(path_parameters.get_base_path(), "computation_parameters", path_parameters.relative_path))
        {
            json computation_params = read_parameter_file(path_parameters.get_base_path(), "computation_parameters", path_parameters.relative_path);
            computation_parameters = ComputationParameters(computation_params);
        }
        else
        {
            std::cout << "Computations are performed with default computation parameters - change them with the function: set_computation_parameters for your main module parameters" << std::endl;
        }
    }

    void write_to_file(const std::string& results_dir) {
        json output_json = prepare_output_json(results_dir);
        Parameters::write_parameter_file(output_json, path_parameters.get_base_path() + results_dir + "/", "config", path_parameters.relative_path);
    }

    std::string get_mode() const
    {
        return mode;
    }

    std::string get_theory() const
    {
        return path_parameters.theory;
    }

    std::string get_root_dir() const
    {
        return path_parameters.root_dir;
    }

    bool get_relative_path () const
    {
        return path_parameters.relative_path;
    }

    PathParameters get_path_parameters() const
    {
        return path_parameters;
    }

protected:
    const std::string mode;
    PathParameters path_parameters;
    const json global_params;
    bool writing_to_file = false;

    ComputationParameters computation_parameters;

    json prepare_output_json(const std::string results_dir)
    {
        writing_to_file = true;

        std::string path = get_absolute_path(path_parameters.get_base_path(), path_parameters.relative_path);
        // ToDo: Needs to be adapted to work for both scenarios
        // json global_parameters = read_parameter_file(gcp() + "/flow_equations/source/" + path_parameters.theory + "/", "global_parameters", false);
        json global_parameters = read_parameter_file(gcp() + "/flow_equations/" + path_parameters.theory + "/", "global_parameters", false);
        generate_directory_if_not_present(path_parameters.get_base_path()); // Base path
        generate_directory_if_not_present(path_parameters.get_base_path() + results_dir + "/"); // Results dir

        // Override computation parameter file
        computation_parameters.write_to_file(path_parameters.get_base_path(), "computation_parameters", path_parameters.relative_path);

        return subtract(params, global_parameters);
    }
};

#endif //PROGRAM_FRGVISUALIZATION_PARAMETERS_HPP
