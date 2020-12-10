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

#include "../util/header.hpp"
#include "../util/dev_dat.hpp"
#include "../util/monitor.hpp"
#include "../extern/thrust_functors.hpp"
#include "../flow_equation_interface/flow_equation_system.hpp"
#include "leaf.hpp"
#include "gridcomputationwrapper.hpp"
#include "nodesexpander.hpp"

class HyperCubes
{
public:
    // Base constructor
    HyperCubes(const cudaT k_, const std::vector< std::vector<int> > n_branches_per_depth_,
               const std::vector <std::pair<cudaT, cudaT> > lambda_ranges_) :
            dim(lambda_ranges_.size()), k(k_),
            n_branches_per_depth(n_branches_per_depth_),
            accum_n_branches_per_dim(HyperCubes::compute_accum_n_branches_per_dim(n_branches_per_depth_, lambda_ranges_.size())),
            accum_n_branches_per_depth(HyperCubes::compute_accum_n_branches_per_depth(n_branches_per_depth_, lambda_ranges_.size())),
            lambda_ranges(lambda_ranges_)
    {}

    // ToDo: Add rule of five

    static thrust::host_vector< thrust::host_vector<int> > compute_accum_n_branches_per_dim(const std::vector< std::vector<int> > &n_branches_per_depth_, const uint dim_);

    static thrust::host_vector< thrust::host_vector<int> > compute_accum_n_branches_per_depth(const std::vector< std::vector<int> > &n_branches_per_depth_, const uint dim_);

    static void compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs);

    // Main Function for node expansion- Executes extract_node_information and expand_node_information_according_to_number_of_nodes to
    // return expanded_cube_indices_ptr and expanded_depth_per_cube_ptr for further computation
    GridComputationWrapper generate_and_linearize_nodes(const int total_number_of_cubes, const int maximum_depth,
                                                        const std::vector<Node* > &nodes_to_be_computed) const;

    // Further (simpler functions for generating grid computation wrapper from other resources

    // Compute expanded cube indices and expanded depth per cube from coordinates -> result can be used in the next instance to compute vertices on the given grid
    GridComputationWrapper project_coordinates_on_expanded_cube_and_depth_per_cube_indices(DevDatC coordinates, bool coordinates_on_grid=false, int depth=-1) const; // no reference since coordinates is changed within this function

    // Compute expanded cube indices and expanded depth per cube from leaves
    GridComputationWrapper project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<Leaf* > &leaves, int depth=-1) const;

    // Cuda code - Compute vertices based on expanded cube index vectors
    void compute_vertices(GridComputationWrapper &grcompwrap);
    void compute_reference_vertices(GridComputationWrapper &grcompwrap);
    void compute_cube_center_vertices(GridComputationWrapper &grcompwrap);

    void determine_vertex_velocities(FlowEquationsWrapper * flow_equations);

    thrust::host_vector<int> determine_potential_fixed_points();

    const DevDatC& get_vertices() const;
    const DevDatC& get_vertex_velocities() const;

    // Function for testing if project_coordinates_on_expanded_cube_and_depth_per_cube_indices works
    // Todo: To be implemented
    void test_projection();

protected:
    // Constants
    const uint8_t dim;
    const cudaT k;

    const std::vector< std::vector<int> > n_branches_per_depth;
    const thrust::host_vector< thrust::host_vector<int> > accum_n_branches_per_dim;
    const thrust::host_vector< thrust::host_vector<int> > accum_n_branches_per_depth;
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges;

    // Variables defined depending on your usage
    int total_number_of_cubes;
    DevDatC vertices; // (total_number_of_cubes x n_cube) x dim (len = dim) OR total_number_of_cubes x dim (len = dim)
    DevDatC vertex_velocities; // (total_number_of_cubes x n_cube) x dim (len = dim) OR total_number_of_cubes x dim (len = dim)

    // Possible modes -> correspond to different possible usages of hypercubes
    enum VertexMode { CubeVertices, ReferenceVertices, CenterVertices};
    VertexMode vertex_mode;

    // Helper functions
    void compute_reference_vertex_in_dim(DimensionIteratorC &reference_vertices_, GridComputationWrapper &grid_computation_wrapper, int dim_index) const;
};

struct greater_than_zero
{
    template< typename T>
    __host__ __device__
    T operator()(const T &val) const
    {
        return val > 0;
    }
};

struct accumulate_n_branches
{
    accumulate_n_branches(const int dim_index_, int init_) : dim_index(dim_index_), sum(init_)
    {}

    __host__
    int operator()(const std::vector<int> elem)
    {
        sum = sum * elem[dim_index];
        return sum;
    }

    int sum;
    const int dim_index;
};


struct compute_axis_index_of_coor
{
    compute_axis_index_of_coor(const cudaT lambda_offset_, const cudaT delta_lambda_, const int n_branch_per_depth_at_dim_) :
        lambda_offset(lambda_offset_), delta_lambda(delta_lambda_), n_branch_per_depth_at_dim(n_branch_per_depth_at_dim_)
    {}

    __host__ __device__
    int operator()(const cudaT &coordinate)
    {
        return (int((coordinate+lambda_offset)/delta_lambda) % n_branch_per_depth_at_dim);
    }

    const cudaT lambda_offset;
    const cudaT delta_lambda;
    const int n_branch_per_depth_at_dim;
};


struct sum_axis_indices_to_cube_index
{
    sum_axis_indices_to_cube_index(const int accum_n_branch_per_depth_) :
        accum_n_branch_per_depth(accum_n_branch_per_depth_)
    {}

    __host__ __device__
    int operator()(const int &axis_index, const int &cube_index)
    {
        return cube_index + axis_index * accum_n_branch_per_depth;
    }

    const int accum_n_branch_per_depth;
};

/* Remark: Generating an array for accum_n_branches_per_dim in advance as it is done for the other expanded vectors
 * makes no sense since the values do not differ from node to node (only the depth values) (reconsider this argument)
 * -> this is probably not true, only valid argument would be redundancy */

struct compute_depth_vertex_coor_weight
{
    compute_depth_vertex_coor_weight(const int n_branch_per_depth_, const int accum_n_branch_per_depth_,
            const int depth_weight_divisor_, dev_vec_int const& accum_n_branches_per_dim_) :
            n_branch_per_depth(n_branch_per_depth_), accum_n_branch_per_depth(accum_n_branch_per_depth_),
            depth_weight_divisor(depth_weight_divisor_),
            accum_n_branches_per_dim_ptr(thrust::raw_pointer_cast(&accum_n_branches_per_dim_[0]))
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        int cube_index = thrust::get<0>(t);
        int maximum_cube_depth = thrust::get<1>(t) + 1; // = k
        int current_val = thrust::get<2>(t);

        thrust::get<2>(t) += ((cube_index/ accum_n_branch_per_depth) % n_branch_per_depth) * accum_n_branches_per_dim_ptr[maximum_cube_depth] / depth_weight_divisor;
    }

    const int n_branch_per_depth;
    const int accum_n_branch_per_depth;
    const int depth_weight_divisor;
    const int * accum_n_branches_per_dim_ptr;
};


struct compute_inner_vertex_coor
{
    compute_inner_vertex_coor(const int dim_index_) : dim_index(dim_index_)
    {}

    __host__ __device__
    int operator()(const int &ith_inner_index) const
    {
        return int(ith_inner_index/ pow(2, dim_index)) % 2;
    }

    const int dim_index;
};


struct finalize_vertex_computation
{
    finalize_vertex_computation(const cudaT lambda_range_, const cudaT lambda_offset_,
            dev_vec_int const& accum_n_branches_per_dim_) :
            lambda_range(lambda_range_), lambda_offset(lambda_offset_),
            accum_n_branches_per_dim_ptr(thrust::raw_pointer_cast(&accum_n_branches_per_dim_[0]))
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        int reference_vertex = thrust::get<0>(t);
        int maximum_cube_depth  = thrust::get<1>(t) + 1; // = k
        int inner_vertex_coor = thrust::get<2>(t);

        thrust::get<3>(t) = (reference_vertex + inner_vertex_coor) * lambda_range / accum_n_branches_per_dim_ptr[maximum_cube_depth] + lambda_offset;
    }

    const cudaT lambda_range;
    const cudaT lambda_offset;
    const int * accum_n_branches_per_dim_ptr;
};

struct finalize_reference_vertex_computation
{
    finalize_reference_vertex_computation(const cudaT lambda_range_, const cudaT lambda_offset_,
                                dev_vec_int const& accum_n_branches_per_dim_) :
            lambda_range(lambda_range_), lambda_offset(lambda_offset_),
            accum_n_branches_per_dim_ptr(thrust::raw_pointer_cast(&accum_n_branches_per_dim_[0]))
    {}

    __host__ __device__
    cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
    {

        return  reference_vertex * lambda_range / accum_n_branches_per_dim_ptr[maximum_cube_depth + 1] + lambda_offset;
    }

    const cudaT lambda_range;
    const cudaT lambda_offset;
    const int * accum_n_branches_per_dim_ptr;
};

struct finalize_center_vertex_computation
{
    finalize_center_vertex_computation(const cudaT lambda_range_, const cudaT lambda_offset_,
                                dev_vec_int const& accum_n_branches_per_dim_) :
            lambda_range(lambda_range_), lambda_offset(lambda_offset_),
            accum_n_branches_per_dim_ptr(thrust::raw_pointer_cast(&accum_n_branches_per_dim_[0]))
    {}

    __host__ __device__
    cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
    {
        // int reference_vertex = thrust::get<0>(t);
        // int maximum_cube_depth  = thrust::get<1>(t) + 1; // = k

        return ((reference_vertex) * lambda_range / accum_n_branches_per_dim_ptr[maximum_cube_depth + 1] +
                (reference_vertex + 1) * lambda_range / accum_n_branches_per_dim_ptr[maximum_cube_depth + 1]) / 2 + lambda_offset;
    }

    const cudaT lambda_range;
    const cudaT lambda_offset;
    const int * accum_n_branches_per_dim_ptr;
};

// Checks if the given number of positive signs is equal to 0 or to upper bound.
// If this is not the case, the given cube contains definitly no fixed point.
// With status, the previous status is taken into account (if it has been recognized already as no fixed point)
struct check_for_no_fixed_point
{
    check_for_no_fixed_point(const int upper_bound_): upper_bound(upper_bound_)
    {}

    __host__ __device__
    bool operator()(const int &val, const bool& status) const
    {
        return ((val == upper_bound) or (val == 0)) or status;
    }

    const int upper_bound;
};

#endif //MAIN_HYPERCUBES_HPP