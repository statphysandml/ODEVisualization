#ifndef PROGRAM_KMEANS_CLUSTERING_HPP
#define PROGRAM_KMEANS_CLUSTERING_HPP

#include <param_helper/params.hpp>

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <odesolver/util/kmeans.hpp>


namespace odesolver {
    namespace modes {
        struct KMeansClustering : public param_helper::params::Parameters
        {
            explicit KMeansClustering(
                const json params
            );

            static KMeansClustering generate(
                const uint maximum_expected_number_of_clusters,
                const double upper_bound_for_min_distance,
                const uint maximum_number_of_iterations=1000
            );

            std::string name() const
            {
                return "kmeans_clustering";
            }

            devdat::DevDatC eval(const devdat::DevDatC &device_data) const;

            devdat::DevDatC eval(const devdat::DevDatC &device_data, const uint k) const;

            uint maximum_expected_number_of_clusters_;
            double upper_bound_for_min_distance_;
            uint maximum_number_of_iterations_;
        };
    }
}

#endif //PROGRAM_KMEANS_CLUSTERING_HPP