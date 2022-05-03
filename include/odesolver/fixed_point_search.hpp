#ifndef PROJECT_FIXEDPOINTSEARCH_HPP
#define PROJECT_FIXEDPOINTSEARCH_HPP

#include <sys/file.h>
#include <tuple>

#include "util/dev_dat.hpp"
#include "util/monitor.hpp"
#include "hypercubes/node.hpp"
#include "hypercubes/buffer.hpp"
#include "hypercubes/nodesexpander.hpp"
#include "hypercubes/hypercubes.hpp"
#include "hypercubes/leaf.hpp"
#include "util/helper_functions.hpp"
#include "util/ode_visualisation.hpp"
#include "util/json_conversions.hpp"

#include <param_helper/json.hpp>

using json = nlohmann::json;


class FixedPointSearch : public ODEVisualisation
{
public:
    // From config
    explicit FixedPointSearch(
        const json params,
        std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr,
        std::shared_ptr<JacobianEquationWrapper> jacobians_ptr=nullptr,
        const std::string computation_parameters_path=param_helper::proj::project_root()
    );

    // From file
    static FixedPointSearch from_file(
        const std::string rel_config_dir,
        std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr,
        std::shared_ptr<JacobianEquationWrapper> jacobians_ptr=nullptr,
        const std::string computation_parameters_path=param_helper::proj::project_root()
    );

    // From parameters
    static FixedPointSearch from_parameters(
        const int maximum_recursion_depth,
        const std::vector<std::vector<int>> n_branches_per_depth,
        const std::vector<std::pair<cudaT, cudaT>> lambda_ranges,
        std::shared_ptr<FlowEquationsWrapper> flow_equations_ptr,
        std::shared_ptr<JacobianEquationWrapper> jacobians_ptr=nullptr,
        const std::string computation_parameters_path=param_helper::proj::project_root()
    );

    struct ClusterParameters : public param_helper::params::Parameters
    {
        ClusterParameters(const json params);

        ClusterParameters(
            const uint maximum_expected_number_of_clusters,
            const double upper_bound_for_min_distance,
            const uint maximum_number_of_iterations=1000
        );

        std::string name() const
        {
            return "cluster";
        }

        const uint maximum_expected_number_of_clusters_;
        const double upper_bound_for_min_distance_;
        const uint maximum_number_of_iterations_;
    };

    // Main function

    void find_fixed_points_dynamic_memory();

    void find_fixed_points_preallocated_memory();

    void cluster_solutions_to_fixed_points(const uint maximum_expected_number_of_clusters,
                           const double upper_bound_for_min_distance,
                           const uint maximum_number_of_iterations = 1000);
    void cluster_solutions_to_fixed_points_from_parameters(const FixedPointSearch::ClusterParameters cluster_parameters);
    void cluster_solutions_to_fixed_points_from_file();

    // Getter functions

    std::vector<std::shared_ptr<Leaf>> get_solutions();
    odesolver::DevDatC get_fixed_points() const;

    // File interactions

    void write_solutions_to_file(std::string rel_dir) const;
    void load_solutions_from_file(std::string rel_dir);

    void write_fixed_points_to_file(std::string rel_dir) const;
    void load_fixed_points_from_file(std::string rel_dir);

private:
    uint dim_;
    int maximum_recursion_depth_;

    HyperCubes hypercubes_;

    Buffer buffer_;
    std::vector<std::shared_ptr<Leaf>> solutions_;
    odesolver::DevDatC fixed_points_;

    // Iterate over nodes and generate new nodes based on the indices of pot fixed points
    void generate_new_nodes_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_fixed_points, const std::vector<Node*> &nodes);

    void run_gpu_computing_task();
};

std::vector<std::vector<double>> load_fixed_points(std::string rel_dir);

#endif //PROJECT_FIXEDPOINTSEARCH_HPP
