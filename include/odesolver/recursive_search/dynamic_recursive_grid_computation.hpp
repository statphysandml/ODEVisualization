#ifndef PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP
#define PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <flowequations/flow_equation.hpp>
#include <odesolver/collection/buffer.hpp>
#include <odesolver/collection/collection.hpp>
#include <odesolver/collection/collection_expander.hpp>
#include <odesolver/grid_computation/grid_computation.hpp>


namespace odesolver {
    namespace recursivesearch {
        /** @brief Class for dynamic and recursive computations of vertices and
         * hypercubes on a grid. The get_collection_package() and the
         * get_buffer() function serve as an interface to add new collections of
         * hypercubes after each computation round, triggered by next().
         */
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

            virtual void next(devdat::DevDatC &vertices);

            bool finished();

            bool check_status();

            const std::vector<odesolver::collections::Collection*>& get_collection_package();

            odesolver::collections::Buffer& get_buffer();

            odesolver::gridcomputation::GridComputation& get_hypercubes();

        protected:
            int number_of_cubes_per_gpu_call_;
            int maximum_number_of_gpu_calls_;

            VertexMode vertex_mode_;
            odesolver::gridcomputation::GridComputation hypercubes_;
            odesolver::collections::Buffer buffer_;
            int c_;
            
            std::vector<odesolver::collections::Collection*> collection_package_;
            int expected_number_of_cubes_;
            int expected_maximum_depth_;
        };
    }
}

#endif //PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP
