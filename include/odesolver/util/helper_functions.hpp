//
// Created by kades on 5/22/19.
//

#ifndef PROJECT_HELPER_FUNCTIONS_HPP
#define PROJECT_HELPER_FUNCTIONS_HPP

#include "header.hpp"
#include "dev_dat.hpp"
#include "../extern/kmeans.hpp"
// #include "../include/coordinate_operator.hpp"

#include <numeric>
#include <fstream>

// Both following functions are currently not used
// Aligns device vector of size dim(=len) x n into a single dev_vec of dimension n*dim(=len) with order x, y, z, ...
dev_vec* align_device_data(const std::vector< dev_vec*> device_data);

// Reverts align_device_data
std::vector< dev_vec*> reorder_device_data(dev_vec* aligned_device_data, const uint8_t dim);

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


#endif //PROJECT_HELPER_FUNCTIONS_HPP
