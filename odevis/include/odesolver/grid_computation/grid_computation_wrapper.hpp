#ifndef PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP
#define PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>


namespace odesolver {
    namespace gridcomputation {
        struct GridComputationWrapper
        {
            /** @brief Class for managing expanded cubes and a respective computation of cube vertices in different recursive depths of the rescursive search tree
             * 
             * @param maximum_number_of_cubes: Maximum expected number of cubes encoded in the provided collection_packages
             * @param maximum_depth: Maximum expected recursive depth of cubes encoded in the provided collection_packages
             * @param init_depth_val: ToDo!
             *  */ 
            GridComputationWrapper(
                const int maximum_number_of_cubes=0,
                const int maximum_depth=0,
                const cudaT init_depth_val=0
            );

            // For parent cube indices
            devdat::DevDatInt expanded_element_indices_;
            // For depths
            devdat::DevDatInt expanded_depth_per_element_wrapper_;
            devdat::DimensionIteratorInt& expanded_depth_per_element_;

            void print_expanded_vectors();
        };
    }
}

#endif //PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP
