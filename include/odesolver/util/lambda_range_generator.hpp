//
// Created by lukas on 25.09.19.
//

#ifndef PROGRAM_LAMBDA_RANGE_GENERATOR_HPP
#define PROGRAM_LAMBDA_RANGE_GENERATOR_HPP

#include "header.hpp"

// Iterators over given partial lambda ranges and fix lambdas and returns respective lambda ranges for possible gpu calls
struct LambdaRangeGenerator
{
    LambdaRangeGenerator(
            const std::vector<int> &n_branches,
            const std::vector <std::pair<cudaT, cudaT> > &partial_lambda_ranges,
            const std::vector <std::vector <cudaT> > &fix_lambdas
    );

    const std::vector < std::pair<cudaT, cudaT> > next();

    bool finished() const;

    const std::vector<int> &n_branches_;
    const std::vector <std::pair<cudaT, cudaT> > &partial_lambda_ranges_;
    const std::vector <std::vector <cudaT> > &fix_lambdas_;

    std::vector <std::vector <cudaT>::const_iterator> fix_lambdas_iterators_;

    bool reached_end_;
};

#endif //PROGRAM_LAMBDA_RANGE_GENERATOR_HPP
