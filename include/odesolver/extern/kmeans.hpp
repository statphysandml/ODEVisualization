//
// Created by kades on 5/22/19.
//

#ifndef PROJECT_KMEANS_HPP
#define PROJECT_KMEANS_HPP

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "../util/random.hpp"

class KMeans
{
public:
    KMeans(const uint k_, const std::vector< std::vector<double> > data_) :
    k(k_), data(data_), n_data(data.size()), dim(data[0].size())
    {
        random_data_point_generator = std::uniform_int_distribution<int>(0, n_data-1);
    }

    void initialize()
    {
        centers = std::vector < std::vector<double> >(k);
        std::vector<uint> center_ids(k, n_data);
        for(auto i = 0; i < k; i++)
        {
            auto random_idx = random_data_point_generator(gen);
            while(std::find(center_ids.begin(), center_ids.end(), random_idx) != center_ids.end())
            {
                random_idx = random_data_point_generator(gen);
            }
            centers[i] = data[random_idx];
            center_ids[i] = random_idx;
        }
        /* std::cout << "Initial center Ids:";
        std::for_each(center_ids.begin(), center_ids.end(), [] (const uint& center_id) { std::cout << " " << center_id; }); */
    }

    void apply(uint max_iterations=1000)
    {
        initialize();
        auto c = 0;
        while(c < max_iterations)
        {
            // Compute new assignments
            std::vector<uint> pot_new_assignments = expectation_step();
            // No new assignments -> algorithm has converged
            if(pot_new_assignments == assignments) {
                assignments = pot_new_assignments;
                break;
            }
            // Update assigments
            assignments = pot_new_assignments;
            // Update cluster centers
            maximization_step();
            c++;
        }

        if(k == 1)
            maximization_step(); // To compute at least one time the actual center;
    }

    // Reassign data to cluster centers
    std::vector<uint> expectation_step()
    {
        std::vector<uint> new_assignments(n_data);
        for(auto i = 0; i < n_data; i++)
        {
            std::vector<double> euclidean_distances = compute_euclidean_distances(data[i], centers);
            auto argmin_it = std::min_element(euclidean_distances.begin(), euclidean_distances.end());
            // std::cout << "Assign data point " << i << " to cluster " << argmin_it - euclidean_distances.begin() << std::endl;
            new_assignments[i] = argmin_it - euclidean_distances.begin();
        }
        return new_assignments;
    }

    // Update cluster centers
    void maximization_step()
    {
        centers = std::vector< std::vector<double> >(k, std::vector<double> (dim, 0));
        for(auto i = 0; i < n_data; i++)
            std::transform(centers[assignments[i]].begin(), centers[assignments[i]].end(), data[i].begin(), centers[assignments[i]].begin(), std::plus<double> ());
        for(auto j = 0; j < k; j++)
        {
            auto n_center_j = std::count(assignments.begin(), assignments.end(), j);
            // std::cout << "Number of clusters in cluster " << j << ": " << n_center_j << std::endl;
            std::transform(centers[j].begin(), centers[j].end(), centers[j].begin(), [n_center_j] (const double &coor) { return coor/n_center_j; });
        }
    }

    double get_mean_distance_to_cluster_centers()
    {
        double mean = 0.0;
        for(auto i = 0; i < n_data; i++) {
            /* std::cout << "Partial distance " << i << " of center " << assignments[i] << " to vertex:";
            std::for_each(data[i].begin(), data[i].end(), [] (const double& center_id) { std::cout << " " << center_id; });
            std::cout << " is: " << compute_euclidean_distance(data[i], centers[assignments[i]]) << std::endl; */
            mean += compute_euclidean_distance(data[i], centers[assignments[i]]);
        }
        return mean/n_data;
    }

    std::vector < std::vector<double> > get_centers()
    {
        return centers;
    }

    void info()
    {
        std::cout << "\nAveraged distance: " << get_mean_distance_to_cluster_centers() << std::endl;
        std::cout << "Cluster centers:" << std::endl;
        std::vector < std::vector<double> > cluster_centers = get_centers();
        for(auto cluster_center : cluster_centers)
        {
            std::cout<< "\t";
            std::for_each(cluster_center.begin(), cluster_center.end(), [] (const double& coor) { std::cout << " " <<  coor; });
            std::cout << std::endl;
        }
    }

private:
    const uint k;
    const std::vector< std::vector<double> > data;
    const uint n_data;
    const uint dim;

    std::uniform_int_distribution<int> random_data_point_generator;
    std::vector < std::vector<double> > centers;
    std::vector<uint> assignments;

    static std::vector<double> compute_euclidean_distances(const std::vector<double> &dat, const std::vector< std::vector<double> > &centers_)
    {
        std::vector<double> euclidean_distances(centers_.size());
        for(auto i = 0; i < euclidean_distances.size(); i++)
            euclidean_distances[i] = compute_euclidean_distance(dat, centers_[i]);
        return euclidean_distances;
    };

    static double compute_euclidean_distance(const std::vector<double> &dat1, const std::vector<double> &dat2) {
        std::vector<double> differences(dat1.size());
        std::transform(dat1.begin(), dat1.end(), dat2.begin(), differences.begin(), std::minus<double> ());
        return std::sqrt(std::inner_product(differences.begin(), differences.end(), differences.begin(), 0.0));
    }
};

#endif //PROJECT_KMEANS_HPP
