#include "../../include/odesolver/util/lambda_range_generator.hpp"


PartialRanges::PartialRanges(
        const std::vector<int> n_branches,
        const std::vector<std::pair<cudaT, cudaT>> partial_lambda_ranges,
        const std::vector<std::vector<cudaT>> fixed_lambdas
) :
        n_branches_(n_branches), partial_lambda_ranges_(partial_lambda_ranges), fixed_lambdas_(fixed_lambdas)
{
    for(auto &fixed_lambda: fixed_lambdas_)
        fixed_lambdas_iterators_.push_back(fixed_lambda.begin());
    reached_end_ = false;
}

const std::vector<std::pair<cudaT, cudaT>> PartialRanges::next()
{
    std::vector<std::pair<cudaT, cudaT>> lambda_ranges;
    lambda_ranges.reserve(n_branches_.size());

    auto partial_lambda_ranges_iterator = partial_lambda_ranges_.begin();
    auto fixed_lambdas_iterators_iter = fixed_lambdas_iterators_.begin();
    auto fixed_lambdas_index = 0;

    for(auto &n_branch : n_branches_)
    {
        if(n_branch > 1) {
            lambda_ranges.push_back(*partial_lambda_ranges_iterator);
            partial_lambda_ranges_iterator++;

            if(fixed_lambdas_index == 0 && fixed_lambdas_.size() == 0)
            {
                reached_end_ = true;
                // Allow further filling of lambda ranges for n_branches > 1
                continue;
            }
        }
        else
        {
            lambda_ranges.push_back(std::pair<cudaT, cudaT> {*(*fixed_lambdas_iterators_iter), *(*fixed_lambdas_iterators_iter) +  0.1});

            // Iterator of last fix lambda is incremented
            if(fixed_lambdas_iterators_iter + 1 == fixed_lambdas_iterators_.end())
                (*fixed_lambdas_iterators_iter)++;
            // Current iterator is at end: increment child iterator and reset current and consecutive iterators
            if(*(fixed_lambdas_iterators_iter) == fixed_lambdas_[fixed_lambdas_index].end()) {
                // Skip iterators that are already at the end
                int c = 1;
                while(*(fixed_lambdas_iterators_iter - c) + 1 == fixed_lambdas_[fixed_lambdas_index - 1].end() and fixed_lambdas_index > 0)
                {
                    fixed_lambdas_index -= 1;
                    c += 1;
                }

                /* // The while loop imitates this behaviour for arbitrary depth
                if(*(fixed_lambdas_iterators_iter - 1) + 1 != fixed_lambdas[fixed_lambdas_index - 1].end())
                    (*(fixed_lambdas_iterators_iter - 1))++;
                else if(*(fixed_lambdas_iterators_iter - 2) + 1 != fixed_lambdas[fixed_lambdas_index - 2].end()) {
                    (*(fixed_lambdas_iterators_iter - 2))++;
                    fixed_lambdas_index -= 1;
                }
                else {
                    (*(fixed_lambdas_iterators_iter - 3))++;
                    fixed_lambdas_index -= 2;
                }*/

                // Condition for ending generator is reached
                if(fixed_lambdas_index == 0)
                {
                    reached_end_ = true;
                    // Allow further filling of lambda ranges for n_branches > 1
                    continue;
                }

                // Increment next possible child iterator
                (*(fixed_lambdas_iterators_iter - c))++;

                // Reset all consecutive iterators to begin
                for(auto i=fixed_lambdas_index; i < fixed_lambdas_.size(); i++) {
                    fixed_lambdas_iterators_[i] = fixed_lambdas_[i].begin();
                }
            }

            // Increment fix lambdas iterators
            fixed_lambdas_iterators_iter++;
            fixed_lambdas_index += 1;
        }
    }

    /* std::cout << "Latest lambda ranges" << std::endl;
    for(auto &lambd_range : lambda_ranges)
        std::cout << lambd_range.first << ", " << lambd_range.second << std::endl;
    std::cout << std::endl; */

    return lambda_ranges;
}

bool PartialRanges::finished() const
{
    return reached_end_;
}