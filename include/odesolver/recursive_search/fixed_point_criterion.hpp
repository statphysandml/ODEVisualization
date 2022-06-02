#ifndef PROJECT_FIXEDPOINTCRITERION_HPP
#define PROJECT_FIXEDPOINTCRITERION_HPP

#include <sys/file.h>
#include <tuple>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/recursive_search/recursive_search_criterion.hpp>


using json = nlohmann::json;


namespace odesolver {
    namespace recursivesearch {
        struct FixedPointCriterion : RecursiveSearchCriterion
        {
            FixedPointCriterion(uint dim) : dim_(dim)
            {}

            static void compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs);

            thrust::host_vector<int> determine_potential_solutions(odesolver::DevDatC& vertices, odesolver::DevDatC& vertex_velocities) override;

            uint dim_;
        };
    }
}

#endif //PROJECT_FIXEDPOINTCRITERION_HPP
