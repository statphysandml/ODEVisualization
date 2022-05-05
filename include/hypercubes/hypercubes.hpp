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
    HyperCubes(const std::vector<std::vector<int>> n_branches_per_depth={},
               const std::vector<std::pair<cudaT, cudaT>> lambda_ranges={});

    static thrust::host_vector<thrust::host_vector<int>> compute_accum_n_branches_per_dim(const std::vector<std::vector<int>> &n_branches_per_depth, const uint dim);

    static thrust::host_vector<thrust::host_vector<int>> compute_accum_n_branches_per_depth(const std::vector<std::vector<int>> &n_branches_per_depth, const uint dim);

    static void compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs);

    // Functions for generating grid computation wrapper from other resources)

    // Compute expanded cube indices and expanded depth per cube from coordinates
    GridComputationWrapper project_coordinates_on_expanded_cube_and_depth_per_cube_indices(const odesolver::DevDatC &coordinates, bool coordinates_on_grid=false, int depth=-1) const; // no reference since coordinates is changed within this function

    // Compute expanded cube indices and expanded depth per cube from leaves
    GridComputationWrapper project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<Leaf*> &leaves, int depth=-1) const;

    void compute_reference_vertices(odesolver::DevDatC &reference_vertices, GridComputationWrapper &grcompwrap);
    odesolver::DevDatC compute_reference_vertices(GridComputationWrapper &grcompwrap);

    void compute_vertices(odesolver::DevDatC &vertices, GridComputationWrapper &grcompwrap, int maximum_depth=0);
    odesolver::DevDatC compute_vertices(GridComputationWrapper &grcompwrap);

    void compute_cube_center_vertices(odesolver::DevDatC &center_vertices, GridComputationWrapper &grcompwrap);
    odesolver::DevDatC compute_cube_center_vertices(GridComputationWrapper &grcompwrap);

    thrust::host_vector<int> determine_potential_fixed_points(odesolver::DevDatC& vertex_velocities);

    const std::vector<std::vector<int>>& get_n_branches_per_depth() const;

    const std::vector<std::pair<cudaT, cudaT>>& get_lambda_ranges() const;

protected:
    // Constants
    size_t dim_;

    std::vector<std::vector<int>> n_branches_per_depth_;
    thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_dim_;
    thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_depth_;
    std::vector<std::pair<cudaT, cudaT>> lambda_ranges_;

    // Helper functions
    void compute_reference_vertex_in_dim(odesolver::DimensionIteratorC &reference_vertices, GridComputationWrapper &grcompwrap, int dim_index, int maximum_depth=0) const;
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
    accumulate_n_branches(const int dim_index, int init) : dim_index_(dim_index), sum_(init)
    {}

    __host__
    int operator()(const std::vector<int> elem)
    {
        sum_ = sum_ * elem[dim_index_];
        return sum_;
    }

    int sum_;
    const int dim_index_;
};


struct compute_axis_index_of_coor
{
    compute_axis_index_of_coor(const cudaT lambda_offset, const cudaT delta_lambda, const int n_branch_per_depth_at_dim) :
        lambda_offset_(lambda_offset), delta_lambda_(delta_lambda), n_branch_per_depth_at_dim_(n_branch_per_depth_at_dim)
    {}

    __host__ __device__
    int operator()(const cudaT &coordinate)
    {
        return (int((coordinate + lambda_offset_) / delta_lambda_) % n_branch_per_depth_at_dim_);
    }

    const cudaT lambda_offset_;
    const cudaT delta_lambda_;
    const int n_branch_per_depth_at_dim_;
};


struct sum_axis_indices_to_cube_index
{
    sum_axis_indices_to_cube_index(const int accum_n_branch_per_depth) :
        accum_n_branch_per_depth_(accum_n_branch_per_depth)
    {}

    __host__ __device__
    int operator()(const int &axis_index, const int &cube_index)
    {
        return cube_index + axis_index * accum_n_branch_per_depth_;
    }

    const int accum_n_branch_per_depth_;
};


struct compute_depth_vertex_coor_weight
{
    compute_depth_vertex_coor_weight(const int n_branch_per_depth, const int accum_n_branch_per_depth,
            const int depth_weight_divisor, dev_vec_int const &accum_n_branches_per_dim) :
            n_branch_per_depth_(n_branch_per_depth), accum_n_branch_per_depth_(accum_n_branch_per_depth),
            depth_weight_divisor_(depth_weight_divisor),
            accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        int cube_index = thrust::get<0>(t);
        int maximum_cube_depth = thrust::get<1>(t) + 1; // = k
        int current_val = thrust::get<2>(t);

        thrust::get<2>(t) += ((cube_index / accum_n_branch_per_depth_) % n_branch_per_depth_) * accum_n_branches_per_dim_ptr_[maximum_cube_depth] / depth_weight_divisor_;
    }

    const int n_branch_per_depth_;
    const int accum_n_branch_per_depth_;
    const int depth_weight_divisor_;
    const int * accum_n_branches_per_dim_ptr_;
};


struct compute_inner_vertex_coor
{
    compute_inner_vertex_coor(const int dim_index) : dim_index_(dim_index)
    {}

    __host__ __device__
    int operator()(const int &ith_inner_index) const
    {
        return int(ith_inner_index/ pow(2, dim_index_)) % 2;
    }

    const int dim_index_;
};


struct finalize_vertex_computation
{
    finalize_vertex_computation(const cudaT lambda_range, const cudaT lambda_offset,
            dev_vec_int const& accum_n_branches_per_dim) :
            lambda_range_(lambda_range), lambda_offset_(lambda_offset),
            accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        int reference_vertex = thrust::get<0>(t);
        int maximum_cube_depth  = thrust::get<1>(t) + 1; // = k
        int inner_vertex_coor = thrust::get<2>(t);

        thrust::get<3>(t) = (reference_vertex + inner_vertex_coor) * lambda_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth] + lambda_offset_;
    }

    const cudaT lambda_range_;
    const cudaT lambda_offset_;
    const int * accum_n_branches_per_dim_ptr_;
};


struct finalize_reference_vertex_computation
{
    finalize_reference_vertex_computation(const cudaT lambda_range, const cudaT lambda_offset,
                                dev_vec_int const& accum_n_branches_per_dim) :
            lambda_range_(lambda_range), lambda_offset_(lambda_offset),
            accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
    {}

    __host__ __device__
    cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
    {

        return  reference_vertex * lambda_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1] + lambda_offset_;
    }

    const cudaT lambda_range_;
    const cudaT lambda_offset_;
    const int * accum_n_branches_per_dim_ptr_;
};


struct finalize_center_vertex_computation
{
    finalize_center_vertex_computation(const cudaT lambda_range, const cudaT lambda_offset,
                                dev_vec_int const& accum_n_branches_per_dim) :
            lambda_range_(lambda_range), lambda_offset_(lambda_offset),
            accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
    {}

    __host__ __device__
    cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
    {
        return ((reference_vertex) * lambda_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1] +
                (reference_vertex + 1) * lambda_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1]) / 2 + lambda_offset_;
    }

    const cudaT lambda_range_;
    const cudaT lambda_offset_;
    const int * accum_n_branches_per_dim_ptr_;
};


// Checks if the given number of positive signs is equal to 0 or to upper bound.
// If this is not the case, the given cube contains definitely no fixed point.
// With status, the previous status is taken into account (if it has been recognized already as no fixed point)
struct check_for_no_fixed_point
{
    check_for_no_fixed_point(const int upper_bound): upper_bound_(upper_bound)
    {}

    __host__ __device__
    bool operator()(const int &val, const bool& status) const
    {
        return ((val == upper_bound_) or (val == 0)) or status;
    }

    const int upper_bound_;
};


#endif //MAIN_HYPERCUBES_HPP