//
// Created by lukas on 13.03.19.
//

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
#include "util/frgvisualization_parameters.hpp"
#include "util/path_parameters.hpp"
// ## ToDo: Reinclude - Commented during reodering #include "coordinate_operator.hpp"

#include <param_helper/json.hpp>

using json = nlohmann::json;


class FixedPointSearch : public FRGVisualizationParameters
{
public:
    // From config
    explicit FixedPointSearch(const json params, const PathParameters path_parameters);

    // From file
    FixedPointSearch(
        const std::string theory,
        const std::string mode_type,
        const std::string results_dir,
        const std::string root_dir="/data/",
        const bool relative_path=true
    );

    // From parameters
    FixedPointSearch(
            const std::string theory,
            const int maximum_recursion_depth,
            const std::vector< std::vector<int> > n_branches_per_depth,
            const std::vector <std::pair<cudaT, cudaT> > lambda_ranges,
            const std::string mode="fixed_point_search", // default mode
            const std::string root_dir="/data/",
            const bool relative_path=true
    );

    struct ClusterParameters : public Parameters
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

    ~FixedPointSearch()
    {
        clear_solutions();
    }

    // Main function

    void find_fixed_point_solutions();

    void cluster_solutions_to_fixed_points(const uint maximum_expected_number_of_clusters,
                           const double upper_bound_for_min_distance,
                           const uint maximum_number_of_iterations = 1000);
    void cluster_solutions_to_fixed_points_from_parameters(const FixedPointSearch::ClusterParameters cluster_parameters);
    void cluster_solutions_to_fixed_points_from_file();

    // Getter functions

    std::vector< Leaf* > get_solutions();
    odesolver::DevDatC get_fixed_points() const;

    // File interactions

    void compute_and_write_fixed_point_characteristics_to_file(std::string dir);

    void write_solutions_to_file(std::string dir) const;
    void load_solutions_from_file(std::string dir);

    void write_fixed_points_to_file(std::string dir) const;
    void load_fixed_points_from_file(std::string dir);

private:
    uint dim_;
    int maximum_recursion_depth_;
    cudaT k_;

    std::vector< std::vector<int> > n_branches_per_depth_;
    std::vector <std::pair<cudaT, cudaT> > lambda_ranges_;

    FlowEquationsWrapper * flow_equations_;

    Buffer buffer_;
    std::vector<Leaf*> solutions_;
    odesolver::DevDatC fixed_points_;

    // Iterate over nodes and generate new nodes based on the indices of pot fixed points
    std::tuple< std::vector<Node* >, std::vector< Leaf* > > generate_new_nodes_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_fixed_points, const std::vector< Node* > &nodes);

    void run_gpu_computing_task();

    void clear_solutions();
};

#endif //PROJECT_FIXEDPOINTSEARCH_HPP
