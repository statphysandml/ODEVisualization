#include <iostream>
#include <vector>
#include <devdat/header.hpp>
#include <odesolver/util/partial_ranges.hpp>


void partial_ranges_t()
{
    const std::vector<int> n_branches {20, 20, 1, 10, 1};
    const std::vector<std::pair<cudaT, cudaT>> partial_variable_ranges = std::vector<std::pair<cudaT, cudaT>> {std::pair<cudaT, cudaT> (-1.0, 1.0), std::pair<cudaT, cudaT> (0.0, 4.0), std::pair<cudaT, cudaT> (-3.0, 3.0)};
    const std::vector<std::vector<cudaT>> fixed_variables = std::vector<std::vector<cudaT>> {std::vector<cudaT> {0.5, 0.0}, std::vector<cudaT> {0.2, 0.2}, std::vector<cudaT> {0.5, 3.0}};
    odesolver::util::PartialRanges partial_ranges(
        n_branches,
        partial_variable_ranges,
        fixed_variables
    );

    std::vector<std::pair<cudaT, cudaT>> variable_ranges;
    for(auto i = 0; i < partial_ranges.size(); i++)
    {
        auto variable_ranges = partial_ranges[i];
        std::cout << "Variable range" << std::endl;
        for(const auto &variable_range: variable_ranges)
        {
            std::cout << variable_range.first << ", " << variable_range.second << std::endl;
        }
    }
}

void partial_ranges_t2()
{
    const std::vector<int> n_branches {11, 11};
    const std::vector<std::pair<cudaT, cudaT>> partial_variable_ranges = std::vector<std::pair<cudaT, cudaT>> {std::pair<cudaT, cudaT> (-1.0, 1.0), std::pair<cudaT, cudaT> (0.0, 4.0)};
    odesolver::util::PartialRanges partial_ranges(
        n_branches,
        partial_variable_ranges
    );

    std::vector<std::pair<cudaT, cudaT>> variable_ranges;
    for(auto i = 0; i < partial_ranges.size() + 1; i++)
    {
        auto variable_ranges = partial_ranges[i];
        std::cout << "Variable range" << std::endl;
        for(const auto &variable_range: variable_ranges)
        {
            std::cout << variable_range.first << ", " << variable_range.second << std::endl;
        }
    }
}