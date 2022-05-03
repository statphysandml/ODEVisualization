//
// Created by lukas on 26.02.19.
//

#ifndef MAIN_HYPERCUBES_HPP
#define MAIN_HYPERCUBES_HPP

#include <thrust/execution_policy.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <utility>
#include <tuple>
#include <vector>

#include "../odesolver/util/header.hpp"
#include "../odesolver/util/dev_dat.hpp"
#include "../odesolver/util/monitor.hpp"
#include "../odesolver/extern/thrust_functors.hpp"
#include "../flow_equation_interface/flow_equation_system.hpp"
#include "leaf.hpp"
#include "gridcomputationwrapper.hpp"
#include "nodesexpander.hpp"

class HyperCubes
{
public:
    // Base constructor
    HyperCubes(const std::vector< std::vector<int> > n_branches_per_depth_,
               const std::vector <std::pair<cudaT, cudaT> > lambda_ranges_);

    // ToDo: Add rule of five

    static thrust::host_vector< thrust::host_vector<int> > compute_accum_n_branches_per_dim(const std::vector< std::vector<int> > &n_branches_per_depth_, const uint dim_);

    static thrust::host_vector< thrust::host_vector<int> > compute_accum_n_branches_per_depth(const std::vector< std::vector<int> > &n_branches_per_depth_, const uint dim_);

    static void compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs);

    // Main Function for node expansion- Executes extract_node_information and expand_node_information_according_to_number_of_nodes to
    // return expanded_cube_indices_ptr and expanded_depth_per_cube_ptr for further computation
    GridComputationWrapper generate_and_linearize_nodes(const int total_number_of_cubes, const int maximum_depth,
                                                        const std::vector<Node*> &node_package) const;

    // Further (simpler functions for generating grid computation wrapper from other resources

    // Compute expanded cube indices and expanded depth per cube from coordinates -> result can be used in the next instance to compute vertices on the given grid
    GridComputationWrapper project_coordinates_on_expanded_cube_and_depth_per_cube_indices(odesolver::DevDatC coordinates, bool coordinates_on_grid=false, int depth=-1) const; // no reference since coordinates is changed within this function

    // Compute expanded cube indices and expanded depth per cube from leaves
    GridComputationWrapper project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<Leaf*> &leaves, int depth=-1) const;

    // Cuda code - Compute vertices based on expanded cube index vectors
    void compute_vertices(GridComputationWrapper &grcompwrap);
    void compute_reference_vertices(GridComputationWrapper &grcompwrap);
    void compute_cube_center_vertices(GridComputationWrapper &grcompwrap);

    // void determine_vertex_velocities(FlowEquationsWrapper * flow_equations);

    thrust::host_vector<int> determine_potential_fixed_points(odesolver::DevDatC& vertex_velocities);

    const odesolver::DevDatC& get_vertices() const;
    // const odesolver::DevDatC& get_vertex_velocities() const;

    // Function for testing if project_coordinates_on_expanded_cube_and_depth_per_cube_indices works
    // Todo: To be implemented
    void test_projection();

protected:
    // Constants
    const uint8_t dim;

    const std::vector<std::vector<int>> n_branches_per_depth;
    const thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_dim;
    const thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_depth;
    const std::vector<std::pair<cudaT, cudaT>> lambda_ranges;
    
    // Variables defined depending on your usage
    int total_number_of_cubes;
    odesolver::DevDatC vertices; // (total_number_of_cubes x n_cube) x dim (len = dim) OR total_number_of_cubes x dim (len = dim)
    // odesolver::DevDatC vertex_velocities; // (total_number_of_cubes x n_cube) x dim (len = dim) OR total_number_of_cubes x dim (len = dim)

    // Possible modes -> correspond to different possible usages of hypercubes
    enum VertexMode { CubeVertices, ReferenceVertices, CenterVertices};
    VertexMode vertex_mode;

    // Helper functions
    void compute_reference_vertex_in_dim(odesolver::DimensionIteratorC &reference_vertices_, GridComputationWrapper &grid_computation_wrapper, int dim_index) const;
};

#endif //MAIN_HYPERCUBES_HPP