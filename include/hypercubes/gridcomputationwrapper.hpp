#ifndef PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP
#define PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP

#include "../odesolver/util/dev_dat.hpp"


struct GridComputationWrapper
{
    /** @brief Class for managing expanded cubes and a respective computation of cube vertices in different recursive depths of the rescursive search tree
     * 
     * @param maximum_number_of_cubes: Maximum expected number of cubes encoded in the provided node_packages
     * @param maximum_depth: Maximum expected recursive depth of cubes encoded in the provided node_packages
     * @param init_depth_val: ToDo!
     *  */ 
    GridComputationWrapper(
        const int maximum_number_of_cubes=0,
        const int maximum_depth=0,
        const cudaT init_depth_val=0
    );

    // For parent cube indices
    odesolver::DevDatInt expanded_cube_indices_;
    // For depths
    odesolver::DevDatInt expanded_depth_per_cube_wrapper_;
    odesolver::DimensionIteratorInt& expanded_depth_per_cube_;

    void print_expanded_vectors();
};

#endif //PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP
