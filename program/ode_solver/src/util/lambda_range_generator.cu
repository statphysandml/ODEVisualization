#include "../../include/util/lambda_range_generator.hpp"


LambdaRangeGenerator::LambdaRangeGenerator(
        const std::vector<int> &n_branches_,
        const std::vector <std::pair<cudaT, cudaT> > &partial_lambda_ranges_,
        const std::vector <std::vector <cudaT> > &fix_lambdas_
) :
        n_branches(n_branches_), partial_lambda_ranges(partial_lambda_ranges_), fix_lambdas(fix_lambdas_)
{
    for(auto &fix_lambd: fix_lambdas)
        fix_lambdas_iterators.push_back(fix_lambd.begin());
    reached_end = false;
}

const std::vector < std::pair<cudaT, cudaT> > LambdaRangeGenerator::next()
{
    std::vector < std::pair<cudaT, cudaT > > lambda_ranges;
    lambda_ranges.reserve(n_branches.size());

    auto partial_lambda_ranges_iterator = partial_lambda_ranges.begin();
    auto fix_lambdas_iterators_iter = fix_lambdas_iterators.begin();
    auto fix_lambdas_index = 0;

    for(auto &n_branch : n_branches)
    {
        if(n_branch > 1) {
            lambda_ranges.push_back(*partial_lambda_ranges_iterator);
            partial_lambda_ranges_iterator++;

            if(fix_lambdas_index == 0 && fix_lambdas.size() == 0)
            {
                reached_end = true;
                // Allow further filling of lambda ranges for n_branches > 1
                continue;
            }
        }
        else
        {
            lambda_ranges.push_back(std::pair<cudaT, cudaT> {*(*fix_lambdas_iterators_iter), *(*fix_lambdas_iterators_iter) +  0.1});

            // Iterator of last fix lambda is incremented
            if(fix_lambdas_iterators_iter + 1 == fix_lambdas_iterators.end())
                (*fix_lambdas_iterators_iter)++;
            // Current iterator is at end: increment child iterator and reset current and consecutive iterators
            if(*(fix_lambdas_iterators_iter) == fix_lambdas[fix_lambdas_index].end()) {
                // Skip iterators that are already at the end
                int c = 1;
                while(*(fix_lambdas_iterators_iter - c) + 1 == fix_lambdas[fix_lambdas_index - 1].end() and fix_lambdas_index > 0)
                {
                    fix_lambdas_index -= 1;
                    c += 1;
                }

                /* // The while loop imitates this behaviour for arbitrary depth
                if(*(fix_lambdas_iterators_iter - 1) + 1 != fix_lambdas[fix_lambdas_index - 1].end())
                    (*(fix_lambdas_iterators_iter - 1))++;
                else if(*(fix_lambdas_iterators_iter - 2) + 1 != fix_lambdas[fix_lambdas_index - 2].end()) {
                    (*(fix_lambdas_iterators_iter - 2))++;
                    fix_lambdas_index -= 1;
                }
                else {
                    (*(fix_lambdas_iterators_iter - 3))++;
                    fix_lambdas_index -= 2;
                }*/

                // Condition for ending generator is reached
                if(fix_lambdas_index == 0)
                {
                    reached_end = true;
                    // Allow further filling of lambda ranges for n_branches > 1
                    continue;
                }

                // Increment next possible child iterator
                (*(fix_lambdas_iterators_iter - c))++;

                // Reset all consecutive iterators to begin
                for(auto i=fix_lambdas_index; i < fix_lambdas.size(); i++) {
                    fix_lambdas_iterators[i] = fix_lambdas[i].begin();
                }
            }

            // Increment fix lambdas iterators
            fix_lambdas_iterators_iter++;
            fix_lambdas_index += 1;
        }
    }

    /* std::cout << "Latest lambda ranges" << std::endl;
    for(auto &lambd_range : lambda_ranges)
        std::cout << lambd_range.first << ", " << lambd_range.second << std::endl;
    std::cout << std::endl; */

    return lambda_ranges;
}

bool LambdaRangeGenerator::finished() const
{
    return reached_end;
}