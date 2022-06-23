#ifndef PROJECT_FIXEDPOINTCRITERION_HPP
#define PROJECT_FIXEDPOINTCRITERION_HPP

#include <sys/file.h>
#include <tuple>

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/recursive_search/recursive_search_criterion.hpp>


using json = nlohmann::json;


namespace odesolver {
    namespace recursivesearch {
        struct FixedPointCriterion : RecursiveSearchCriterion
        {
            FixedPointCriterion()
            {}

            static void compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs);

            thrust::host_vector<int> determine_potential_solutions(devdat::DevDatC& vertices, devdat::DevDatC& vertex_velocities) override;
        };
    }
}

#endif //PROJECT_FIXEDPOINTCRITERION_HPP
