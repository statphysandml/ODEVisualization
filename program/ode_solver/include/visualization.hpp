//
// Created by kades on 7/16/19.
//

#ifndef PROJECT_VISUALIZATION_HPP
#define PROJECT_VISUALIZATION_HPP

#include <sys/file.h>
#include <tuple>
#include <fstream>

#include "util/monitor.hpp"
#include "hypercubes/node.hpp"
#include "hypercubes/buffer.hpp"
#include "hypercubes/nodesexpander.hpp"
#include "hypercubes/hypercubes.hpp"
#include "util/helper_functions.hpp"
#include "observers/evolution.hpp"
#include "coordinate_operator.hpp"
#include "observers/conditional_intersection_observer.hpp"

#include "param_helper/pathfinder.hpp"
#include "util/dev_dat.hpp"
#include "util/frgvisualization_parameters.hpp"
#include "util/random.hpp"
#include "util/lambda_range_generator.hpp"
#include "param_helper/json.hpp"
#include "param_helper/params.hpp"

using json = nlohmann::json;

/* Examples:
 * - Full lambda scan
 * - Scan for unique fixed lambdas in two dimensions
 * - Scan for fixed lambdas in two dimensions with one altering value (one fix lambda)
 * - Scan for fixed lambdas in two dimensions with two altering values (two fix lambdas)
 */

class VisualizationParameters : public FRGVisualizationParameters {
public:
    explicit VisualizationParameters(const json params_, const PathParameters path_parameters_);

    // From file
    VisualizationParameters(const std::string theory,
                            const std::string mode_type,
                            const std::string results_dir,
                            const std::string root_dir="/data/",
                            const bool relative_path=true);

    VisualizationParameters(
        const std::string theory_,
        const std::vector<int> n_branches_,
        const std::vector <std::pair<cudaT, cudaT> > lambda_ranges_,
        const std::vector <std::vector <cudaT> > fix_lambdas_ = std::vector< std::vector <cudaT> > {},
        const std::string mode_="visualization", // default mode
        const std::string root_dir="/data/",
        const bool relative_path=true
        );

    void append_explicit_points_parameters(
            std::vector< std::vector<cudaT> > explicit_points=std::vector< std::vector<cudaT> >{},
            std::string source_root_dir="/data/",
            bool source_relative_path=true,
            std::string source_file_dir="");

    struct ComputeVertexVelocitiesParameters : public Parameters
    {
        ComputeVertexVelocitiesParameters(const json params_);

        ComputeVertexVelocitiesParameters(
                const bool skip_fix_lambdas_, const bool with_vertices_
        );

        std::string name() const {  return "compute_vertex_velocities";  }

        const bool skip_fix_lambdas;
        const bool with_vertices;
    };

    struct ComputeSeparatrizesParameters : public Parameters
    {
        ComputeSeparatrizesParameters(const json params_);

        ComputeSeparatrizesParameters(
                const uint N_per_eigen_dim_,
                const std::vector<double> shift_per_dim_
        );

        std::string name() const {  return "compute_separatrizes";  }

        const uint N_per_eigen_dim;
        const std::vector<double> shift_per_dim;
    };

private:
    friend class Visualization;

    const uint dim;
    const cudaT k;

    std::vector<int> n_branches;
    std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges;
    std::vector <std::vector <cudaT> > fix_lambdas;

    FlowEquationsWrapper * flow_equations;

    std::vector< std::vector<cudaT> > get_fixed_points() const;
};


class Visualization
{
public:
    explicit Visualization(const VisualizationParameters &vp_);

    void compute_vertex_velocities(std::string dir, bool skip_fix_lambdas, bool with_vertices);

    void compute_vertex_velocities_from_parameters(std::string dir);

    void compute_separatrizes(const std::string dir,
                              const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges,
                              const std::vector <cudaT> minimum_change_of_state,
                              const cudaT minimum_delta_t, const cudaT maximum_flow_val,
                              const std::vector <cudaT> vicinity_distances,
                              const uint observe_every_nth_step, const uint maximum_total_number_of_steps,
                              const uint N_per_eigen_dim,
                              const std::vector<double> shift_per_dim);

    void compute_separatrizes_from_parameters(const std::string dir);

private:
    const VisualizationParameters &vp;

    std::vector<int> indices_of_fixed_lambdas;

    HyperCubes * compute_vertex_velocities_of_sub_problem(
            const int number_of_cubes_per_gpu_call,
            Buffer * buffer_ptr,
            const std::vector <std::pair<cudaT, cudaT> > lambda_ranges);

    DevDatC sample_around_saddle_point(const std::vector<double> coordinate, const std::vector<int> manifold_indices,
                                       const std::vector<std::vector<cudaT>> manifold_eigenvectors, const std::vector<double> shift_per_dim, const uint N_per_eigen_dim);

    DevDatC get_initial_values_to_eigenvector(const std::vector<double> saddle_point, std::vector<cudaT> eigenvector, const std::vector<double> shift_per_dim);

    void extract_stable_and_unstable_manifolds(
            Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType eigenvalue,
            Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType eigenvector,
            std::vector<int> &stable_manifold_indices,
            std::vector<int> &unstable_manifold_indices,
            std::vector<std::vector<cudaT>> &manifold_eigenvectors
    );

    // ToDo: Check for https://www.boost.org/doc/libs/1_72_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/tutorial/self_expanding_lattices.html to obtain faster results
    void compute_separatrizes_of_manifold(
            const std::vector<double> saddle_point,
            const std::vector<int> manifold_indices,
            const std::vector<std::vector<cudaT>> manifold_eigenvectors,
            const cudaT delta_t,
            const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges,
            const std::vector <cudaT> minimum_change_of_state,
            const cudaT minimum_delta_t, const cudaT maximum_flow_val,
            const std::vector <cudaT> vicinity_distances,
            const uint observe_every_nth_step, const uint maximum_total_number_of_steps,
            const uint N_per_eigen_dim,
            const std::vector<double> shift_per_dim,
            std::ofstream &os,
            std::vector< cudaT > fixed_lambdas
    );
};

#endif //PROJECT_VISUALIZATION_HPP
