#ifndef PROGRAM_HYPERCUBES_HPP
#define PROGRAM_HYPERCUBES_HPP

#include <thrust/execution_policy.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <utility>
#include <tuple>
#include <vector>

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <devdat/util/thrust_functors.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/collection/leaf.hpp>
#include <odesolver/collection/collection.hpp>
#include <odesolver/collection/collection_expander.hpp>
#include <odesolver/grid_computation/grid_computation_wrapper.hpp>


namespace odesolver {
    namespace gridcomputation {
        class GridComputation
        {
        public:
            // Base constructor
            GridComputation(const std::vector<std::vector<int>> n_branches_per_depth={},
                    const std::vector<std::pair<cudaT, cudaT>> variable_ranges={});

            static thrust::host_vector<thrust::host_vector<int>> compute_accum_n_branches_per_dim(const std::vector<std::vector<int>> &n_branches_per_depth, const uint dim);

            static thrust::host_vector<thrust::host_vector<int>> compute_accum_n_branches_per_depth(const std::vector<std::vector<int>> &n_branches_per_depth, const uint dim);

            // Functions for generating grid computation wrapper from other resources)

            // Compute expanded cube indices and expanded depth per cube from coordinates
            GridComputationWrapper project_coordinates_on_expanded_cube_and_depth_per_cube_indices(const devdat::DevDatC &coordinates, bool coordinates_on_grid=false, int depth=-1) const;
            
            // Compute expanded cube indices and expanded depth per cube from leaves
            GridComputationWrapper project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<odesolver::collections::Leaf*> &leaves, int depth=-1) const;

            // Compute expanded cube indices and expanded depth per cube from collections
            GridComputationWrapper  project_collection_package_on_expanded_cube_and_depth_per_cube_indices(std::vector<odesolver::collections::Collection*> &collection_package, int expected_number_of_cubes, int expected_maximum_depth_) const;

            void compute_reference_vertices(devdat::DevDatC &reference_vertices, GridComputationWrapper &grcompwrap, int maximum_depth=0);
            devdat::DevDatC compute_reference_vertices(GridComputationWrapper &grcompwrap);

            void compute_cube_vertices(devdat::DevDatC &cube_vertices, GridComputationWrapper &grcompwrap, int maximum_depth=0);
            devdat::DevDatC compute_cube_vertices(GridComputationWrapper &grcompwrap);

            void compute_cube_center_vertices(devdat::DevDatC &center_vertices, GridComputationWrapper &grcompwrap, int maximum_depth=0);
            devdat::DevDatC compute_cube_center_vertices(GridComputationWrapper &grcompwrap);

            const std::vector<std::vector<int>> n_branches_per_depth() const;

            const std::vector<std::pair<cudaT, cudaT>> variable_ranges() const;

            size_t dim() const;

        protected:
            // Constants
            size_t dim_;

            std::vector<std::vector<int>> n_branches_per_depth_;
            thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_dim_;
            thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_depth_;
            std::vector<std::pair<cudaT, cudaT>> variable_ranges_;

            // Helper functions
            void compute_reference_vertex_in_dim(devdat::DimensionIteratorC &reference_vertices, GridComputationWrapper &grcompwrap, int dim_index, int maximum_depth=0) const;
        };
    }
}

#endif //PROGRAM_HYPERCUBES_HPP
