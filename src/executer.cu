#include "../include/executer.hpp"

#include <fstream>

const std::map<std::string, Executer::ExecutionMode> Executer::mode_resolver = {
    {"visualization", visualization},
    {"fixed_point_search", fixed_point_search},
    {"evaluate", evaluate},
    {"jacobian", jacobian},
    {"evolve", evolve},
    {"evolve_with_conditional_range_observer", evolve_with_conditional_range_observer},
    {"evolve_with_conditional_intersection_observer", evolve_with_conditional_range_observer},
};


void Executer::exec_visualization(const VisualizationParameters& visualization_parameters, std::string dir)
{
    Visualization visualization(visualization_parameters);
    visualization.compute_flow_from_parameters(dir);

    auto path_parameters_ = visualization_parameters.get_path_parameters();

    // visualization.compute_separatrizes_from_parameters(dir);

    FILE* file;
    auto args = prepare_file("visualisation", dir, path_parameters_);
    PySys_SetArgv(args.first, args.second);
    std::cout << gcp() << std::endl;
    std::fstream fileStream;
    // ToDo: Path to ode_solver must somehow be important - for example by a config file or a bash script that tells you the correct name
    file = fopen((gcp() + "/../../program/plotting_routines/modes/visualisation.py").c_str(),"r");
    PyRun_SimpleFile(file, "visualisation.py");
    fclose(file);

    /* FILE* file;

    Py_Initialize();
    // PySys_SetArgv(2, argv);
    PyRun_SimpleString("import sys\n" "import os");
    PyRun_SimpleString("sys.path.append( os.path.dirname(os.getcwd()) +'/./plotting_routines/')");
    PyRun_SimpleString("print(os.path.dirname(os.getcwd()) +'/./plotting_routines/')");
    PyRun_SimpleString("print(sys.version)");
    // PyRun_SimpleString("import numpy as np");
    file = fopen("./../plotting_routines/main.py","r");
    PyRun_SimpleFile(file, "main.py");
    Py_Finalize();
    break; */
}

void Executer::prep_exec_visualization(std::string dir, const PathParameters path_parameters)
{
    const std::string theory = path_parameters.theory;

    auto * flow_equation = FlowEquationsWrapper::make_flow_equation(path_parameters.theory);
    auto dim = flow_equation->get_dim();

    const int initial_n_branches = 20;
    std::vector<int> n_branches (dim, 1);
    n_branches[0] = initial_n_branches;
    n_branches[1] = initial_n_branches;

    const std::vector <std::pair<cudaT, cudaT> > partial_variable_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-1.0, 1.0), std::pair<cudaT, cudaT> (-1.0, 1.0)};

    std::vector <std::vector <cudaT> > fixed_variables = {};
    if(dim > 2)
        fixed_variables = std::vector< std::vector<cudaT> >(
                dim - 2, std::vector<cudaT>{-0.5, 0.0, 0.5});

    VisualizationParameters visualization_parameters(theory, n_branches, partial_variable_ranges, fixed_variables);

    std::vector <std::vector <cudaT> > explicit_points = {std::vector<cudaT> (dim, 0.5), std::vector<cudaT> (dim, -0.5)};

    visualization_parameters.append_explicit_points_parameters(explicit_points);
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fixed_variables = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_flow_parameters(skip_fixed_variables, with_vertices);
    visualization_parameters.append_parameters(compute_flow_parameters);

    visualization_parameters.write_to_file(dir);
}

void Executer::exec_fixed_point_search(const FixedPointSearchParameters& fixed_point_search_parameters, std::string dir)
{
    FixedPointSearch fixed_point_search(fixed_point_search_parameters);

    fixed_point_search.find_fixed_point_solutions();
    fixed_point_search.write_solutions_to_file(dir);

    Counter<Collection>::print_statistics();

    // Cluster solutions
    fixed_point_search.cluster_solutions_to_fixed_points_from_file();
    fixed_point_search.write_fixed_points_to_file(dir);
    fixed_point_search.compute_and_write_fixed_point_characteristics_to_file(dir);
}

void Executer::prep_exec_fixed_point_search(std::string dir, const PathParameters path_parameters)
{
    const std::string theory = path_parameters.theory;

    auto * flow_equation = FlowEquationsWrapper::make_flow_equation(path_parameters.theory);
    auto dim = flow_equation->get_dim();

    const int maximum_recursion_depth = 8;
    const int n_branches_in_most_upper_depth = 20;
    std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> >(
            maximum_recursion_depth, std::vector<int> (dim, 2));
    std::transform(n_branches_per_depth[0].begin(), n_branches_per_depth[0].end(), n_branches_per_depth[0].begin(),
            [n_branches_in_most_upper_depth] (const int& val) { return n_branches_in_most_upper_depth; });

    std::vector <std::pair<cudaT, cudaT> > variable_ranges = std::vector <std::pair<cudaT, cudaT> > (
            dim, std::pair<cudaT, cudaT> (-1.0, 1.0));

    FixedPointSearchParameters fixed_point_search_parameters(
            theory,
            maximum_recursion_depth,
            n_branches_per_depth,
            variable_ranges);

    // Setting gpu specific computation parameters (optional) - parameters are already set default
    const int number_of_cubes_per_gpu_call = 20000;
    const int maximum_number_of_gpu_calls = 1000;
    fixed_point_search_parameters.set_computation_parameters(
            number_of_cubes_per_gpu_call,
            maximum_number_of_gpu_calls);

    // Parameters for clustering the resulting solutions - Represent parameters of a function
    const uint maximum_expected_number_of_clusters = 80;
    const double upper_bound_for_min_distance = 0.01;
    const uint maximum_number_of_iterations = 1000;
    FixedPointSearchParameters::ClusterParameters cluster_parameters(maximum_expected_number_of_clusters,
                                                                     upper_bound_for_min_distance, maximum_number_of_iterations);
    fixed_point_search_parameters.append_parameters(cluster_parameters);

    fixed_point_search_parameters.write_to_file(dir);
}

void Executer::exec_evaluate(const CoordinateOperatorParameters& evaluation_parameters, std::string dir)
{
    CoordinateOperator evaluator(evaluation_parameters);

    evaluator.compute_velocities();
    evaluator.write_characteristics_to_file(dir);
}

void Executer::prep_exec_evaluate(std::string dir, const PathParameters path_parameters)
{
    const std::string theory = path_parameters.theory;

    auto * flow_equation = FlowEquationsWrapper::make_flow_equation(path_parameters.theory);
    auto dim = flow_equation->get_dim();

    std::vector< std::vector<cudaT> > coordinates = std::vector< std::vector<cudaT> >(2, std::vector<cudaT> (dim, 0.0));

    CoordinateOperatorParameters evaluator_parameters = CoordinateOperatorParameters::generate(
            theory, coordinates, "evaluate");
    evaluator_parameters.write_to_file(dir);
}

void Executer::exec_jacobian(const CoordinateOperatorParameters& evaluation_parameters, std::string dir)
{
    CoordinateOperator evaluator(evaluation_parameters);

    // Compute velocities only if necessary!!
    evaluator.compute_velocities();
    evaluator.compute_jacobians_and_eigendata();

    evaluator.write_characteristics_to_file(dir);
}

void Executer::prep_exec_jacobian(std::string dir, const PathParameters path_parameters)
{
    const std::string theory = path_parameters.theory;

    auto * flow_equation = FlowEquationsWrapper::make_flow_equation(path_parameters.theory);
    auto dim = flow_equation->get_dim();

    std::vector< std::vector<cudaT> > coordinates = std::vector< std::vector<cudaT> >(2, std::vector<cudaT> (dim, 0.0));

    CoordinateOperatorParameters evaluator_parameters = CoordinateOperatorParameters::generate(
            theory, coordinates, "jacobian");
    evaluator_parameters.write_to_file(dir);
}

void Executer::exec_evolve(const CoordinateOperatorParameters& evolve_parameters, std::string dir)
{
    CoordinateOperator evolve(evolve_parameters);
    Fileos fileos(evolve_parameters.get_path_parameters().get_base_path() + dir + "/");
    TrackingObserver observer(fileos.get());
    evolve.evolve_from_file(observer);
}

void Executer::exec_evolve_with_conditional_range_observer(const CoordinateOperatorParameters& evolve_parameters, std::string dir)
{
    CoordinateOperator evolve(evolve_parameters);
    // evolve.evolve_on_condition_from_file<ConditionalRangeObserverParameters>();
}

void Executer::exec_evolve_with_conditional_intersection_observer(const CoordinateOperatorParameters& evolve_parameters, std::string dir)
{
    CoordinateOperator evolve(evolve_parameters);
    // evolve.evolve_on_condition_from_file<ConditionalIntersectionObserverParameters>();
}


void run_from_file(int argc, char **argv)
{
    print_system_info();

    // ./ODESolver {theory_name} {execution_mode} {output_ordner}

    std::string theory = std::string(argv[1]);
    std::string mode_type = std::string(argv[2]);
    std::string mode_dir = std::string(argv[3]);
    std::string data_root_dir = "/data/";
    bool relative_path = true;
    if(argc > 4) {
        data_root_dir = std::string(argv[4]);
        relative_path = false;
    }
    std::cout << "theory: " << theory << std::endl;
    std::cout << "mode_type: " << mode_type << std::endl;
    std::cout << "mode_dir: " << mode_dir << std::endl;
    std::cout << "data_root_dir" << data_root_dir << std::endl;
    std::cout << "rp: " << relative_path << std::endl;

    PathParameters path_parameters(theory, mode_type, data_root_dir, relative_path);
    Executer executer(path_parameters);
    executer.main(mode_dir);
}