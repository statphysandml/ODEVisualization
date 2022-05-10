#ifndef PROGRAM_HYPERCUBES_HPP
#define PROGRAM_HYPERCUBES_HPP

#include <thrust/execution_policy.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <utility>
#include <tuple>
#include <vector>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/thrust_functors.hpp>
#include <odesolver/flow_equations/flow_equation_system.hpp>
#include <odesolver/collection/leaf.hpp>
#include <odesolver/grid_computation/grid_computation_wrapper.hpp>


namespace odesolver {
    namespace grid_computation {
        class GridComputation
        {
        public:
            // Base constructor
            GridComputation(const std::vector<std::vector<int>> n_branches_per_depth={},
                    const std::vector<std::pair<cudaT, cudaT>> lambda_ranges={});

            static thrust::host_vector<thrust::host_vector<int>> compute_accum_n_branches_per_dim(const std::vector<std::vector<int>> &n_branches_per_depth, const uint dim);

            static thrust::host_vector<thrust::host_vector<int>> compute_accum_n_branches_per_depth(const std::vector<std::vector<int>> &n_branches_per_depth, const uint dim);

            // Functions for generating grid computation wrapper from other resources)

            // Compute expanded cube indices and expanded depth per cube from coordinates
            GridComputationWrapper project_coordinates_on_expanded_cube_and_depth_per_cube_indices(const odesolver::DevDatC &coordinates, bool coordinates_on_grid=false, int depth=-1) const; // no reference since coordinates is changed within this function

            // Compute expanded cube indices and expanded depth per cube from leaves
            GridComputationWrapper project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<odesolver::collections::Leaf*> &leaves, int depth=-1) const;

            void compute_reference_vertices(odesolver::DevDatC &reference_vertices, GridComputationWrapper &grcompwrap, int maximum_depth=0);
            odesolver::DevDatC compute_reference_vertices(GridComputationWrapper &grcompwrap);

            void compute_vertices(odesolver::DevDatC &vertices, GridComputationWrapper &grcompwrap, int maximum_depth=0);
            odesolver::DevDatC compute_vertices(GridComputationWrapper &grcompwrap);

            void compute_cube_center_vertices(odesolver::DevDatC &center_vertices, GridComputationWrapper &grcompwrap, int maximum_depth=0);
            odesolver::DevDatC compute_cube_center_vertices(GridComputationWrapper &grcompwrap);

            const std::vector<std::vector<int>>& get_n_branches_per_depth() const;

            const std::vector<std::pair<cudaT, cudaT>>& get_lambda_ranges() const;

            size_t dim() const;

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
    }
}

#endif //PROGRAM_HYPERCUBES_HPP
