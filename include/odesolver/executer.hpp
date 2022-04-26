//
// Created by lukas on 31.07.19.
//

#ifndef PROGRAM_EXECUTER_HPP
#define PROGRAM_EXECUTER_HPP

#include "util/frgvisualization_parameters.hpp"
#include "fixed_point_search.hpp"
#include "visualization.hpp"
#include "coordinate_operator.hpp"

#include "../../examples/tests.hpp"

#include <iostream>
#include <map>
#include <Python.h>
#include <stdio.h>

#include "param_helper/json.hpp"

using json = nlohmann::json;

struct Executer
{
    enum ExecutionMode {visualization, fixed_point_search, evaluate, jacobian, evolve,
            evolve_with_conditional_range_observer, evolve_with_conditional_intersection_observer};
    
    static const std::map< std::string, ExecutionMode> mode_resolver;

    Executer(const PathParameters path_parameters_) : path_parameters(path_parameters_)
    {}

    static std::string mode_to_string (ExecutionMode mode_) {
        switch (mode_) {
            case visualization:
                return "visualization";
            case fixed_point_search:
                return "fixed_point_search";
            case evaluate:
                return "evaluate";
            case jacobian:
                return "jacobian";
            case evolve:
                return "evolve";
            case evolve_with_conditional_range_observer:
                return "evolve_with_conditional_range_observer";
            case evolve_with_conditional_intersection_observer:
                return "evolve_with_conditional_intersection_observer";
            default:
                return "mode_not_known";
        }
    }

    static void exec_visualization(const VisualizationParameters& visualization_parameters, std::string dir);
    static void prep_exec_visualization(std::string dir, const PathParameters path_parameters);
    static void exec_fixed_point_search(const FixedPointSearchParameters& fixed_point_search_parameters, std::string dir);
    static void prep_exec_fixed_point_search(std::string dir, const PathParameters path_parameters);
    static void exec_evaluate(const CoordinateOperatorParameters& evaluation_parameters, std::string dir);
    static void prep_exec_evaluate(std::string dir, const PathParameters path_parameters);
    static void exec_jacobian(const CoordinateOperatorParameters& evaluation_parameters, std::string dir);
    static void prep_exec_jacobian(std::string dir, const PathParameters path_parameters);
    static void exec_evolve(const CoordinateOperatorParameters& evolve_parameters, std::string dir);
    // static void prep_exec_evolve(std::string dir);
    static void exec_evolve_with_conditional_range_observer(const CoordinateOperatorParameters& evolve_parameters, std::string dir);
    // static void prep_exec_evolve_with_conditional_range_observer(std::string dir);
    static void exec_evolve_with_conditional_intersection_observer(const CoordinateOperatorParameters& evolve_parameters, std::string dir);
    // static void prep_exec_evolve_with_conditional_intersection_observer(std::string dir);

    void main(std::string dir)
    {
        ExecutionMode mode;
        json config_file;
        bool in_preparation;
        if(Parameters::check_if_parameter_file_exists(path_parameters.get_base_path() + "/" + dir + "/", "config", path_parameters.relative_path))
        {
            std::cout << "Mode = " << path_parameters.mode_type << " will be executed based on provided config.json file" << std::endl;
            config_file = Parameters::read_parameter_file(path_parameters.get_base_path() + "/" + dir + "/", "config", path_parameters.relative_path);
            mode = mode_resolver.at(config_file["mode"].get<std::string>());
            in_preparation = false;
            if(config_file["mode"].get<std::string>() != path_parameters.mode_type)
            {
                std::cout << "\nERROR: Mode in config file " << config_file["mode"].get<std::string>()
                          << " and mode in execution command " << path_parameters.mode_type
                          << " do not coincide" << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        else
        {
            std::cout << "A default config.json file will be generated in " << dir
                      << ". Adapt it to your considered set of equations and run the execution command again."
                      << std::endl;
            mode = mode_resolver.at(path_parameters.mode_type);
            in_preparation = true;
        }

        switch (mode) {
            case visualization:
            {
                if(in_preparation)
                    prep_exec_visualization(dir, path_parameters);
                else
                {
                    VisualizationParameters visualization_parameters(config_file, path_parameters);
                    exec_visualization(visualization_parameters, dir);
                }
                break;
            }
            case fixed_point_search:
            {
                if(in_preparation)
                    prep_exec_fixed_point_search(dir, path_parameters);
                else
                {
                    FixedPointSearchParameters fixed_point_search_parameters(config_file, path_parameters);
                    exec_fixed_point_search(fixed_point_search_parameters, dir);
                }
                break;
            }
            case evaluate:
            {
                if(in_preparation)
                    prep_exec_evaluate(dir, path_parameters);
                else
                {
                    CoordinateOperatorParameters evaluation_parameters(config_file, path_parameters);
                    exec_evaluate(evaluation_parameters, dir);
                }
                break;
            }
            case jacobian:
            {
                if(in_preparation)
                    prep_exec_jacobian(dir, path_parameters);
                else
                {
                    CoordinateOperatorParameters evaluation_parameters(config_file, path_parameters);
                    exec_jacobian(evaluation_parameters, dir);
                }
                break;
            }
            case evolve:
            {
                CoordinateOperatorParameters evolve_parameters(config_file, path_parameters);
                exec_evolve(evolve_parameters, dir);
                break;
            }
            case evolve_with_conditional_range_observer:
            {
                CoordinateOperatorParameters evolve_parameters(config_file, path_parameters);
                exec_evolve_with_conditional_range_observer(evolve_parameters, dir);
                break;
            }
            case evolve_with_conditional_intersection_observer:
            {
                CoordinateOperatorParameters evolve_parameters(config_file, path_parameters);
                exec_evolve_with_conditional_intersection_observer(evolve_parameters, dir);
                break;
            }
            default:
                std::cout << "mode not known..." << std::endl;
                break;
        }
    }

    static std::pair<int, wchar_t**> prepare_file(std::string python_file, std::string dir, PathParameters path_parameters_)
    {
        int argc = 5;
        const char * argv[5];

        argv[0] = python_file.c_str();
        argv[1] = path_parameters_.theory.c_str();
        std::cout << "Path: " << path_parameters_.get_base_path() << "\t.\t" << path_parameters_.get_base_path().c_str() << std::endl;
        auto project_directory = gcp();
        project_directory.erase (project_directory.end()-6, project_directory.end());
        argv[2] = project_directory.c_str();
        argv[3] = dir.c_str();
        argv[4] = bool_to_string(path_parameters_.relative_path).c_str();
        auto** _argv = (wchar_t**) PyMem_Malloc(sizeof(wchar_t*)*argc);
        for (int i=0; i<argc; i++) {
            wchar_t* arg = Py_DecodeLocale(argv[i], NULL);
            _argv[i] = arg;
        }

        return std::pair<int, wchar_t**> {argc, _argv};
    }

    inline static std::string bool_to_string(bool b)
    {
        return b ? "true" : "false";
    }

    const PathParameters path_parameters;
};


void run_from_file(int argc, char **argv);


#endif //PROGRAM_EXECUTER_HPP
