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
    HyperCubes(const std::vector<std::vector<int>> n_branches_per_depth_={},
               const std::vector<std::pair<cudaT, cudaT>> lambda_ranges_={});

    static thrust::host_vector<thrust::host_vector<int>> compute_accum_n_branches_per_dim(const std::vector<std::vector<int>> &n_branches_per_depth_, const uint dim_);

    static thrust::host_vector<thrust::host_vector<int>> compute_accum_n_branches_per_depth(const std::vector<std::vector<int>> &n_branches_per_depth_, const uint dim_);

    static void compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs);

    // Further (simpler functions for generating grid computation wrapper from other resources)

    // Compute expanded cube indices and expanded depth per cube from coordinates -> result can be used in the next instance to compute vertices on the given grid
    GridComputationWrapper project_coordinates_on_expanded_cube_and_depth_per_cube_indices(odesolver::DevDatC coordinates, bool coordinates_on_grid=false, int depth=-1) const; // no reference since coordinates is changed within this function

    // Compute expanded cube indices and expanded depth per cube from leaves
    GridComputationWrapper project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<Leaf*> &leaves, int depth=-1) const;

    // Cuda code - Compute vertices based on expanded cube index vectors

    void compute_reference_vertices(odesolver::DevDatC &reference_vertices, GridComputationWrapper &grcompwrap);
    odesolver::DevDatC compute_reference_vertices(GridComputationWrapper &grcompwrap);

    void compute_vertices(odesolver::DevDatC &vertices, GridComputationWrapper &grcompwrap, int maximum_depth=0);
    odesolver::DevDatC compute_vertices(GridComputationWrapper &grcompwrap);

    void compute_cube_center_vertices(odesolver::DevDatC &center_vertices, GridComputationWrapper &grcompwrap);
    odesolver::DevDatC compute_cube_center_vertices(GridComputationWrapper &grcompwrap);

    // void determine_vertex_velocities(FlowEquationsWrapper * flow_equations);

    thrust::host_vector<int> determine_potential_fixed_points(odesolver::DevDatC& vertex_velocities);

    // const odesolver::DevDatC& get_vertices() const;
    // const odesolver::DevDatC& get_vertex_velocities() const;

    const std::vector<std::vector<int>>& get_n_branches_per_depth() const;

    const std::vector<std::pair<cudaT, cudaT>>& get_lambda_ranges() const;

    // Function for testing if project_coordinates_on_expanded_cube_and_depth_per_cube_indices works
    // Todo: To be implemented
    void test_projection();

protected:
    // Constants
    size_t dim;

    std::vector<std::vector<int>> n_branches_per_depth;
    thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_dim;
    thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_depth;
    std::vector<std::pair<cudaT, cudaT>> lambda_ranges;

    // Variables defined depending on your usage
    // odesolver::DevDatC vertices; // (total_number_of_cubes x n_cube) x dim (len = dim) OR total_number_of_cubes x dim (len = dim)
    // odesolver::DevDatC vertex_velocities; // (total_number_of_cubes x n_cube) x dim (len = dim) OR total_number_of_cubes x dim (len = dim)

    // Possible modes -> correspond to different possible usages of hypercubes
    enum VertexMode { CubeVertices, ReferenceVertices, CenterVertices};
    VertexMode vertex_mode;

    // Helper functions
    void compute_reference_vertex_in_dim(odesolver::DimensionIteratorC &reference_vertices_, GridComputationWrapper &grcompwrap, int dim_index, int maximum_depth=0) const;
};

#endif //MAIN_HYPERCUBES_HPP