#ifndef PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP
#define PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/computation_parameters.hpp>
#include <odesolver/flow_equations/flow_equation.hpp>
#include <odesolver/collection/buffer.hpp>
#include <odesolver/collection/collection.hpp>
#include <odesolver/collection/collection_expander.hpp>
#include <odesolver/grid_computation/grid_computation.hpp>


namespace odesolver {
    namespace gridcomputation {
        class DynamicRecursiveGridComputation
        {
        public:
            // Possible modes -> correspond to different possible usages of hypercubes
            enum VertexMode { CenterVertices, CubeVertices, ReferenceVertices};

            DynamicRecursiveGridComputation(
                const int number_of_cubes_per_gpu_call=20000,
                const int maximum_number_of_gpu_calls=1000
            );

            void initialize(
                const std::vector<std::vector<int>> n_branches_per_depth,
                const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
                VertexMode vertex_mode
            );

            virtual void next(odesolver::DevDatC &vertices);

            bool finished();

            bool check_status();

            const std::vector<odesolver::collections::Collection*>& get_collection_package();

            odesolver::collections::Buffer& get_buffer();

            GridComputation& get_hypercubes();

        protected:
            int number_of_cubes_per_gpu_call_;
            int maximum_number_of_gpu_calls_;

            VertexMode vertex_mode_;
            GridComputation hypercubes_;
            odesolver::collections::Buffer buffer_;
            int c_;
            
            std::vector<odesolver::collections::Collection*> collection_package_;
            int expected_number_of_cubes_;
            int expected_maximum_depth_;
        };
    }
}

#endif //PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP
