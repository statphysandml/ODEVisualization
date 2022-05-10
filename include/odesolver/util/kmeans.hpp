//
// Created by kades on 5/22/19.
//

#ifndef PROJECT_KMEANS_HPP
#define PROJECT_KMEANS_HPP

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <odesolver/util/random.hpp>


namespace odesolver {
    namespace util {
        class KMeans
        {
        public:
            KMeans(const uint k, const std::vector< std::vector<double> > data) ;

            void initialize();

            void apply(uint max_iterations=1000);

            // Reassign data to cluster centers
            std::vector<uint> expectation_step();

            // Update cluster centers
            void maximization_step();

            double get_mean_distance_to_cluster_centers();

            std::vector < std::vector<double> > get_centers();

            void info();

        private:
            const uint k_;
            const std::vector< std::vector<double> > data_;
            const uint n_data_;
            const uint dim_;

            std::uniform_int_distribution<int> random_data_point_generator_;
            std::vector < std::vector<double> > centers_;
            std::vector<uint> assignments_;

            static std::vector<double> compute_euclidean_distances(const std::vector<double> &dat, const std::vector< std::vector<double> > &centers);

            static double compute_euclidean_distance(const std::vector<double> &dat1, const std::vector<double> &dat2);
        };
    }
}

#endif //PROJECT_KMEANS_HPP
