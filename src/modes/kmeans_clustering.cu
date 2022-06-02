#include <odesolver/modes/kmeans_clustering.hpp>

namespace odesolver {
    namespace modes {
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

        KMeansClustering::KMeansClustering(
            const json params
        ) : Parameters(params),
            maximum_expected_number_of_clusters_(get_entry<uint>("maximum_expected_number_of_clusters")),
            upper_bound_for_min_distance_(get_entry<double>("upper_bound_for_min_distance")),
            maximum_number_of_iterations_(get_entry<uint>("maximum_number_of_iterations"))
        {}

        KMeansClustering KMeansClustering::generate(
            const uint maximum_expected_number_of_clusters,
            const double upper_bound_for_min_distance,
            const uint maximum_number_of_iterations
        )
        {
            return KMeansClustering(
                json {{"maximum_expected_number_of_clusters", maximum_expected_number_of_clusters},
                    {"upper_bound_for_min_distance", upper_bound_for_min_distance},
                    {"maximum_number_of_iterations", maximum_number_of_iterations}}
            );
        }
        

        odesolver::DevDatC KMeansClustering::eval(const odesolver::DevDatC &device_data) const
        {
            auto transposed_data = device_data.to_vec_vec();
            auto maximum_k = std::min(maximum_expected_number_of_clusters_, uint(transposed_data.size()));

            // Perform clustering for different values of k
            std::vector<double> averaged_distances(maximum_k);
            for(auto k = 1; k <= maximum_k; k++)
            {
                odesolver::util::KMeans kmeans(k, transposed_data);
                kmeans.apply(maximum_number_of_iterations_);
                std::cout << "Averaged distance for k = " << k << ": " << kmeans.get_mean_distance_to_cluster_centers() << std::endl;
                averaged_distances[k-1] = kmeans.get_mean_distance_to_cluster_centers();
            }

            // A comparison of the inner distance of clusters and the distance to the mean is another possibility to check

            // Get knee position
            std::vector<double> adjacent_differences(maximum_k);
            std::adjacent_difference(averaged_distances.begin(), averaged_distances.end(), adjacent_differences.begin());
            std::cout << "Adjacent differences:";
            std::for_each(adjacent_differences.begin(), adjacent_differences.end(), [] (const double & center_id) { std::cout << " " << center_id; });
            std::cout << std::endl;

            uint appropriate_k = 1;
            auto maximum_diff = adjacent_differences[0];
            std::cout << "Maximum diff k: " << maximum_diff << std::endl;
            for(auto i = 1; i < adjacent_differences.size(); i++)
            {
                if((averaged_distances[i] < upper_bound_for_min_distance_ and appropriate_k == 1) or // first time below the threshold
                (averaged_distances[i] < upper_bound_for_min_distance_ and std::abs(adjacent_differences[i]) >  maximum_diff))
                {
                    appropriate_k = i + 1;
                    maximum_diff = std::abs(adjacent_differences[i]);
                }
            }

            /* if(adjacent_differences.size() > 1)
            {
                auto argmax_it = std::max_element(adjacent_differences.begin() + 1, adjacent_differences.end(), abs_compare(upper_bound_for_min_distance_, averaged_distances));
                std::cout << "\nArgmax points to" << *argmax_it << std::endl;
                appropriate_k = argmax_it - adjacent_differences.begin() + 1;
            }
            else
                appropriate_k = 1;
            std::cout << "Appropriate k " << appropriate_k << std::endl;*/

            // ToDo: Plot result

            // Perform clustering ones more to get the cluster centers
            return eval(device_data, appropriate_k);
        }

        odesolver::DevDatC KMeansClustering::eval(const odesolver::DevDatC &device_data, const uint k) const
        {
            auto transposed_data = device_data.to_vec_vec();
            odesolver::util::KMeans kmeans(k, transposed_data);
            kmeans.apply(maximum_number_of_iterations_);
            std::cout << "Final averaged distance: " << kmeans.get_mean_distance_to_cluster_centers() << " for k=" << k << std::endl;
            std::vector<std::vector<double>> cluster_centers = kmeans.get_centers();
            for(auto cluster_center : cluster_centers)
            {
                std::cout << "Cluster center";
                for(auto elem : cluster_center)
                    std::cout << " " << elem;
                std::cout << std::endl;
            }
            return odesolver::DevDatC(cluster_centers);
        }
    }
}