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
#include "coordinate_operator.hpp"

#include "param_helper/json.hpp"

using json = nlohmann::json;


class FixedPointSearchParameters : public FRGVisualizationParameters {
public:
    // From config
    explicit FixedPointSearchParameters(const json params_, const PathParameters path_parameters_);

    // From file
    FixedPointSearchParameters(const std::string theory,
                               const std::string mode_type,
                               const std::string results_dir,
                               const std::string root_dir="/data/",
                               const bool relative_path=true);

    // From parameters
    FixedPointSearchParameters(
            const std::string theory_,
            const int maximum_recursion_depth_,
            const std::vector< std::vector<int> > n_branches_per_depth_,
            const std::vector <std::pair<cudaT, cudaT> > lambda_ranges_,
            const std::string mode_="fixed_point_search", // default mode
            const std::string root_dir="/data/",
            const bool relative_path=true
    );

    struct ClusterParameters : public Parameters
    {
        ClusterParameters(const json params_);

        ClusterParameters(
                const uint maximum_expected_number_of_clusters_,
                const double upper_bound_for_min_distance_,
                const uint maximum_number_of_iterations_=1000
        );

        std::string name() const
        {
            return "cluster";
        }

        const uint maximum_expected_number_of_clusters;
        const double upper_bound_for_min_distance;
        const uint maximum_number_of_iterations;
    };

private:
    friend class FixedPointSearch;

    const uint dim;
    const int maximum_recursion_depth;
    const cudaT k;

    std::vector< std::vector<int> > n_branches_per_depth;
    std::vector <std::pair<cudaT, cudaT> > lambda_ranges;

    FlowEquationsWrapper * flow_equations;
};


class FixedPointSearch
{
public:
    FixedPointSearch(const FixedPointSearchParameters &sp_);

    ~FixedPointSearch()
    {
        clear_solutions();
    }

    // Main function

    void find_fixed_point_solutions();

    void cluster_solutions_to_fixed_points(const uint maximum_expected_number_of_clusters,
                           const double upper_bound_for_min_distance,
                           const uint maximum_number_of_iterations = 1000);
    void cluster_solutions_to_fixed_points_from_parameters(const FixedPointSearchParameters::ClusterParameters cluster_parameters);
    void cluster_solutions_to_fixed_points_from_file();

    // Getter functions

    std::vector< Leaf* > get_solutions();
    DevDatC get_fixed_points() const;

    // File interactions

    void compute_and_write_fixed_point_characteristics_to_file(std::string dir);

    void write_solutions_to_file(std::string dir) const;
    void load_solutions_from_file(std::string dir);

    void write_fixed_points_to_file(std::string dir) const;
    void load_fixed_points_from_file(std::string dir);

private:
    const FixedPointSearchParameters &sp;

    Buffer buffer;
    std::vector< Leaf* > solutions;
    DevDatC fixed_points;

    // Iterate over nodes and generate new nodes based on the indices of pot fixed points
    std::tuple< std::vector<Node* >, std::vector< Leaf* > > generate_new_nodes_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_fixed_points, const std::vector< Node* > &nodes);

    void run_gpu_computing_task();

    void clear_solutions();
};

#endif //PROJECT_FIXEDPOINTSEARCH_HPP
