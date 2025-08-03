#ifndef PROGRAM_STATIC_RECURSIVE_GRID_COMPUTATION_HPP
#define PROGRAM_STATIC_RECURSIVE_GRID_COMPUTATION_HPP

#include <odesolver/recursive_search/dynamic_recursive_grid_computation.hpp>


namespace odesolver {
    namespace recursivesearch {
        class StaticRecursiveGridComputation : public DynamicRecursiveGridComputation
        {
        public:
            StaticRecursiveGridComputation(
                const int maximum_recursion_depth,
                const int number_of_cubes_per_gpu_call=20000,
                const int maximum_number_of_gpu_calls=1000
            );

            void next(devdat::DevDatC &vertices) override;

        private:
            odesolver::gridcomputation::GridComputationWrapper grid_computation_wrapper_;
        };
    }
}

#endif //PROGRAM_STATIC_RECURSIVE_GRID_COMPUTATION_HPP
