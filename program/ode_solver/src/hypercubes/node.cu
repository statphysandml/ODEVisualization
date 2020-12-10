#include "../../include/hypercubes/node.hpp"

int compute_internal_end_index(const std::vector<int> &n_branches)
{
    return std::accumulate(n_branches.begin(), n_branches.end(), 1, std::multiplies<int>());
}