//
// Created by lukas on 25.09.19.
//

#ifndef PROGRAM_LAMBDA_RANGE_GENERATOR_HPP
#define PROGRAM_LAMBDA_RANGE_GENERATOR_HPP

#include "header.hpp"

struct PartialRanges
{
    /** @brief Helper class allowing for the preparation of a finite grid on a hyperplane, characterised by n_branches, partial_lambda_ranges and fixed_lambdas.
     * @param n_branches: Number of branches per dimension. Dimensions with a fixed lambda are supposed to be indicated with a 1, for example, {10, 1, 10}, with the second dimension referring to a fixed lambda
     * @param partial_lambda_ranges: Parameter ranges of the non-fixed lambdas
     * @param fixed_lambdas: List of vectors of fixed lambdas determining the position of the hyperplane in the higher-dimensional space. For each vector of this list, respective lambda_ranges for all dimensions are generated by the next() function
     */
    PartialRanges(
        const std::vector<int> n_branches,
        const std::vector<std::pair<cudaT, cudaT>> partial_lambda_ranges,
        const std::vector<std::vector<cudaT>> fixed_lambdas
    );

    size_t size() const
    {
        return fixed_lambdas_.size();
    }

    std::vector<std::pair<cudaT, cudaT>> operator[] (int i) const
    {
        return dimension_iterators_[i];
    }


    std::vector<int> n_branches_;
    std::vector<std::pair<cudaT, cudaT>> partial_lambda_ranges_;
    std::vector<std::vector<cudaT>> fixed_lambdas_;
};

#endif //PROGRAM_LAMBDA_RANGE_GENERATOR_HPP
