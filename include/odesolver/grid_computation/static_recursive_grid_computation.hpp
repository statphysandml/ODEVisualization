#ifndef PROGRAM_STATIC_RECURSIVE_GRID_COMPUTATION_HPP
#define PROGRAM_STATIC_RECURSIVE_GRID_COMPUTATION_HPP

#include <odesolver/grid_computation/dynamic_recursive_grid_computation.hpp>


namespace odesolver {
    namespace gridcomputation {
        class StaticRecursiveGridComputation : public DynamicRecursiveGridComputation
        {
        public:
            StaticRecursiveGridComputation(
                const int maximum_recursion_depth,
                const int number_of_cubes_per_gpu_call=20000,
                const int maximum_number_of_gpu_calls=1000
            );

            void next(odesolver::DevDatC &vertices) override;

        private:
            GridComputationWrapper grid_computation_wrapper_;
        };
    }
}

#endif //PROGRAM_STATIC_RECURSIVE_GRID_COMPUTATION_HPP
