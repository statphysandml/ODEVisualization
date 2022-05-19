#include <odesolver/util/partial_ranges.hpp>


namespace odesolver {
    namespace util {
        PartialRanges::PartialRanges(
                const std::vector<int> n_branches,
                const std::vector<std::pair<cudaT, cudaT>> partial_variable_ranges,
                const std::vector<std::vector<cudaT>> fixed_variables
        ) :
                n_branches_(n_branches), partial_variable_ranges_(partial_variable_ranges), fixed_variables_(fixed_variables)
        {
            for (const auto &fixed_variable : fixed_variables_)
            {
                if(fixed_variable.size() + partial_variable_ranges.size() != n_branches.size()) {
                    std::cout << "\nERROR: Number of coordinates for at least one vector in fixed_variables is not consistent with the expected dimension defined by n_branches." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        size_t PartialRanges::size() const
        {
            return fixed_variables_.size();
        }

        std::vector<std::pair<cudaT, cudaT>> PartialRanges::operator[] (int i) const
        {
            std::vector<std::pair<cudaT, cudaT>> variable_ranges;
            variable_ranges.reserve(n_branches_.size());
            
            auto partial_variable_ranges_iterator = partial_variable_ranges_.begin();
            auto fixed_variables_iterator = fixed_variables_[i].begin();
            for(auto &n_branch : n_branches_)
            {
                if(n_branch > 1) {
                    variable_ranges.push_back(*partial_variable_ranges_iterator);
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
