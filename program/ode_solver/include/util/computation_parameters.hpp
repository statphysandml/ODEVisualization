//
// Created by lukas on 31.03.20.
//

#ifndef PROGRAM_COMPUTATION_PARAMETERS_HPP
#define PROGRAM_COMPUTATION_PARAMETERS_HPP

#include "param_helper/params.hpp"

class ComputationParameters : public Parameters {
public:
    explicit ComputationParameters(const json params_) : Parameters(params_),
                                                         number_of_cubes_per_gpu_call(get_value_by_key<int>(
                                                                 "number_of_cubes_per_gpu_call")),
                                                         maximum_number_of_gpu_calls(get_value_by_key<int>(
                                                                 "maximum_number_of_gpu_calls"))
    {}
    ComputationParameters(const int number_of_cubes_per_gpu_call_=20000,
                          const int maximum_number_of_gpu_calls_=1000) : ComputationParameters(
            json{{"number_of_cubes_per_gpu_call", number_of_cubes_per_gpu_call_},
                 {"maximum_number_of_gpu_calls",  maximum_number_of_gpu_calls_}}
    )
    {}

    int number_of_cubes_per_gpu_call;
    int maximum_number_of_gpu_calls;
};

#endif //PROGRAM_COMPUTATION_PARAMETERS_HPP
