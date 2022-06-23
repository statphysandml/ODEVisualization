#ifndef PROJECT_RECURSIVESEARCHCRITERION_HPP
#define PROJECT_RECURSIVESEARCHCRITERION_HPP

#include <sys/file.h>
#include <tuple>

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <odesolver/util/monitor.hpp>


using json = nlohmann::json;


namespace odesolver {
    namespace recursivesearch {
        struct RecursiveSearchCriterion
        {
            virtual thrust::host_vector<int> determine_potential_solutions(devdat::DevDatC& vertices, devdat::DevDatC& vertex_velocities) = 0;
        };
    }
}

#endif //PROJECT_RECURSIVESEARCHCRITERION_HPP
