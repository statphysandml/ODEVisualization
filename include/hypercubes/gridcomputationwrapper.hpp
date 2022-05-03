//
// Created by lukas on 23.09.19.
//

#ifndef PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP
#define PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP

#include "../odesolver/util/dev_dat.hpp"

struct GridComputationWrapper
{
    GridComputationWrapper(const int total_number_of_cubes, const int maximum_depth, const cudaT init_depth_val = 0);

    void print_expanded_vectors();

    // For parent cube indices
    odesolver::DevDatInt expanded_cube_indices;
    // For depths
    odesolver::DevDatInt expanded_depth_per_cube_wrapper;
    odesolver::DimensionIteratorInt &expanded_depth_per_cube;
};

#endif //PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP
