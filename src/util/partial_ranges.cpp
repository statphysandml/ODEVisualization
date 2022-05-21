#include <odesolver/util/partial_ranges.hpp>


namespace odesolver {
    namespace util {
        PartialRanges::PartialRanges(
                const std::vector<int> n_branches,
                const std::vector<std::pair<cudaT, cudaT>> partial_variable_ranges,
                const std::vector<std::vector<cudaT>> fixed_variables,
                const bool extend
        ) :
                n_branches_(n_branches),
                partial_variable_ranges_(partial_variable_ranges), fixed_variables_(fixed_variables),
                extend_(extend)
        {
            for (const auto &fixed_variable : fixed_variables_)
            {
                if(fixed_variable.size() + partial_variable_ranges.size() != n_branches.size()) {
                    std::cout << "\nERROR: (Number of entries for at least one vector in fixed_variables) + variable_ranges.size() is inconsistent with the n_branches.size()." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        size_t PartialRanges::size() const
        {
            if(fixed_variables_.size() == 0)
                return 1;
            else
                return fixed_variables_.size();
        }

        std::vector<std::pair<cudaT, cudaT>> PartialRanges::operator[] (int i) const
        {
            std::vector<std::pair<cudaT, cudaT>> variable_ranges;
            variable_ranges.reserve(n_branches_.size());
            
            auto partial_variable_ranges_iterator = partial_variable_ranges_.begin();
            std::vector<cudaT>::const_iterator fixed_variables_iterator;
            if(fixed_variables_.size() > 0)
                fixed_variables_iterator = fixed_variables_[i].begin();
            for(auto &n_branch : n_branches_)
            {
                if(n_branch > 1) {
                    if(extend_)
                    {
                        auto del = (partial_variable_ranges_iterator->second - partial_variable_ranges_iterator->first) / (n_branch -1);
                        variable_ranges.push_back(std::pair<cudaT, cudaT>(partial_variable_ranges_iterator->first, partial_variable_ranges_iterator->second + del));
                    }
                    else
                    {
                        variable_ranges.push_back(*partial_variable_ranges_iterator);
                    }
                    partial_variable_ranges_iterator++;
                }
                else {
                    variable_ranges.push_back(std::pair<cudaT, cudaT>{*fixed_variables_iterator, *fixed_variables_iterator});
                    fixed_variables_iterator++;
                }
            }
            return variable_ranges;
        }
    }
}
