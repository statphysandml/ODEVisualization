//
// Created by lukas on 31.03.20.
//

#ifndef PROGRAM_VISUALISATION_HPP
#define PROGRAM_VISUALISATION_HPP

#include "../ode_solver/include/visualization.hpp"

// Mode: "visualization"
void generate_visualization_parameters()
{
    const std::string theory = "three_point_system";
    const std::vector<int> n_branches {20, 20, 1};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.3, 0.9), std::pair<cudaT, cudaT> (-1.0, 0.4)};
    const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > { std::vector<cudaT> {0.0187308, 0, -0.164471} };

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges, fix_lambdas);

    // For example, for fixed_points
    visualization_parameters.append_explicit_points_parameters(std::vector < std::vector<cudaT> > {std::vector<cudaT> {0.299241730826242, -0.508756268365042, 0.0187309537615095}, std::vector<cudaT> {0.570243002084585, -0.164092840047983, -0.164467312739446}, std::vector<cudaT> {6.29425e-08, -1.71661e-07, 3.43323e-07 }});
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    /* const uint N_per_eigen_dim = 20;
    const std::vector<double> shift_per_dim {0.001, 0.001, 0.01};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
        std::pair<cudaT, cudaT> (-0.3, 0.9), std::pair<cudaT, cudaT> (-1.0, 0.4), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-10, 1e-10, 1e-10};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    const std::vector <cudaT > vicinity_dimensions = std::vector< cudaT > {1e-2};
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val, vicinity_dimensions);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 200;
    const uint maximum_total_number_of_steps = 1000000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_on_condition_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_on_condition_parameters); */

    visualization_parameters.write_to_file("visualization");
}


#endif //PROGRAM_VISUALISATION_WITH_PARAMS_FILES_HPP
