#include <odesolver/util/partial_ranges.hpp>


namespace odesolver {
    namespace util {
        PartialRanges::PartialRanges(
                const std::vector<int> n_branches,
                const std::vector<std::pair<cudaT, cudaT>> partial_lambda_ranges,
                const std::vector<std::vector<cudaT>> fixed_lambdas
        ) :
                n_branches_(n_branches), partial_lambda_ranges_(partial_lambda_ranges), fixed_lambdas_(fixed_lambdas)
        {
            for (const auto &fixed_lambda : fixed_lambdas_)
            {
                if(fixed_lambda.size() + partial_lambda_ranges.size() != n_branches.size()) {
                    std::cout << "\nERROR: Number of coordinates for at least one vector in fixed_lambdas is not consistent with the expected dimension defined by n_branches." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        size_t PartialRanges::size() const
        {
            return fixed_lambdas_.size();
        }

        std::vector<std::pair<cudaT, cudaT>> PartialRanges::operator[] (int i) const
        {
            std::vector<std::pair<cudaT, cudaT>> lambda_ranges;
            lambda_ranges.reserve(n_branches_.size());
            
            auto partial_lambda_ranges_iterator = partial_lambda_ranges_.begin();
            auto fixed_lambdas_iterator = fixed_lambdas_[i].begin();
            for(auto &n_branch : n_branches_)
            {
                if(n_branch > 1) {
                    lambda_ranges.push_back(*partial_lambda_ranges_iterator);
                    partial_lambda_ranges_iterator++;
                }
                else {
                    lambda_ranges.push_back(std::pair<cudaT, cudaT>{*fixed_lambdas_iterator, *fixed_lambdas_iterator});
                    fixed_lambdas_iterator++;
                }
            }
            return lambda_ranges;
        }
    }
}
