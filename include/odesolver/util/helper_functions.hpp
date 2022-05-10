#ifndef PROJECT_HELPER_FUNCTIONS_HPP
#define PROJECT_HELPER_FUNCTIONS_HPP

#include <numeric>
#include <fstream>
#include <vector>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/kmeans.hpp>


namespace odesolver {
    namespace util {
        struct abs_compare
        {
            abs_compare(const double upper_bound_for_min_distance_, const std::vector<double> averaged_distances_) :
                upper_bound_for_min_distance(upper_bound_for_min_distance_), averaged_distances(averaged_distances_)
            {
                i = 1;
            }

            bool operator() (double current_smallest, double a) {
                i = i + 1;
                std::cout << i << ": " << averaged_distances[i] << ", " << a << ", smallest " << current_smallest << std::endl;
                return (std::abs(a) < std::abs(current_smallest)) or (averaged_distances[i] < upper_bound_for_min_distance);
            }

            uint i;
            const double upper_bound_for_min_distance;
            const std::vector<double> averaged_distances;
        };

        odesolver::DevDatC cluster_device_data(
                uint maximum_expected_number_of_clusters,
                double upper_bound_for_min_distance,
                odesolver::DevDatC device_data,
                uint maximum_number_of_iterations=1000);
    }
}

#endif //PROJECT_HELPER_FUNCTIONS_HPP
