#include <iostream>
#include <vector>
#include<odesolver/util/header.hpp>
#include <odesolver/util/partial_ranges.hpp>


void partial_ranges_t()
{
    const std::vector<int> n_branches {20, 20, 1, 10, 1};
    const std::vector<std::pair<cudaT, cudaT>> partial_lambda_ranges = std::vector<std::pair<cudaT, cudaT>> {std::pair<cudaT, cudaT> (-1.0, 1.0), std::pair<cudaT, cudaT> (0.0, 4.0), std::pair<cudaT, cudaT> (-3.0, 3.0)};
    const std::vector<std::vector<cudaT>> fix_lambdas = std::vector<std::vector<cudaT>> {std::vector<cudaT> {0.5, 0.0}, std::vector<cudaT> {0.2, 0.2}, std::vector<cudaT> {0.5, 3.0}};
    PartialRanges partial_ranges(
        n_branches,
        partial_lambda_ranges,
        fix_lambdas
    );

    std::vector<std::pair<cudaT, cudaT>> lambda_ranges;
    for(auto i = 0; i < partial_ranges.size(); i++)
    {
        auto lambda_ranges = partial_ranges[i];
        std::cout << "Lambda range" << std::endl;
        for(const auto &lambda_range: lambda_ranges)
        {
            std::cout << lambda_range.first << ", " << lambda_range.second << std::endl;
        }
    }
}