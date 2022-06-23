#ifndef PROGRAM_MESH_HPP
#define PROGRAM_MESH_HPP

#include <param_helper/params.hpp>

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <devdat/util/json_conversions.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/partial_ranges.hpp>
#include <odesolver/collection/collection.hpp>
#include <odesolver/grid_computation/grid_computation.hpp>


namespace odesolver {
    namespace modes {
        struct Mesh : public param_helper::params::Parameters
        {
            // From config
            explicit Mesh(
                const json params
            );

            // From parameters
            static Mesh generate(
                const std::vector<int> n_branches,
                const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
                const std::vector<std::vector<cudaT>> fixed_variables = std::vector<std::vector<cudaT>> {}
            );

            // From file
            static Mesh from_file(
                const std::string rel_config_dir
            );

            devdat::DevDatC eval(int fixed_variable_idx);

            std::vector<int> n_branches_;
            std::vector<std::pair<cudaT, cudaT>> partial_variable_ranges_;
            std::vector<std::vector<cudaT>> fixed_variables_;
            odesolver::util::PartialRanges partial_ranges_;

            std::vector<int> indices_of_fixed_variables_;
        };
    }
}

#endif //PROGRAM_MESH_HPP
