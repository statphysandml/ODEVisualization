#ifndef PROJECT_RECURSIVESEARCHCRITERION_HPP
#define PROJECT_RECURSIVESEARCHCRITERION_HPP

#include <sys/file.h>
#include <tuple>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>


using json = nlohmann::json;


namespace odesolver {
    namespace recursivesearch {
        struct RecursiveSearchCriterion
        {
            virtual thrust::host_vector<int> determine_potential_solutions(odesolver::DevDatC& vertices, odesolver::DevDatC& vertex_velocities) = 0;
        };
    }
}

#endif //PROJECT_RECURSIVESEARCHCRITERION_HPP
