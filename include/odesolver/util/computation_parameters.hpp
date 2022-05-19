#ifndef PROGRAM_COMPUTATION_PARAMETERS_HPP
#define PROGRAM_COMPUTATION_PARAMETERS_HPP

#include <param_helper/params.hpp>


namespace odesolver {
    namespace util {
        struct ComputationParameters : public param_helper::params::Parameters {
            explicit ComputationParameters(const json params):
                Parameters(params),
                number_of_cubes_per_gpu_call_(get_entry<int>("number_of_cubes_per_gpu_call", 20000)),
                maximum_number_of_gpu_calls_(get_entry<int>("maximum_number_of_gpu_calls", 1000))
            {}
            
            ComputationParameters(
                const int number_of_cubes_per_gpu_call=20000,
                const int maximum_number_of_gpu_calls=1000
            ):
                ComputationParameters(
                    json{{"number_of_cubes_per_gpu_call", number_of_cubes_per_gpu_call},
                        {"maximum_number_of_gpu_calls",  maximum_number_of_gpu_calls}}
                )
            {}

            int number_of_cubes_per_gpu_call_;
            int maximum_number_of_gpu_calls_;
        };
    }
}

#endif //PROGRAM_COMPUTATION_PARAMETERS_HPP
