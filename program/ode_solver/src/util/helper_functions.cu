#include "../../include/util/helper_functions.hpp"


// Aligns device vector of size dim(=len) x n into a single dev_vec of dimension n*dim(=len) with order x, y, z, ...
dev_vec* align_device_data(const std::vector< dev_vec*> device_data)
{
    auto aligned_device_data_ptr = new dev_vec(device_data.size() * device_data[0]->size());
    for(auto dim_index = 0; dim_index < device_data.size(); dim_index ++) {
        thrust::copy(device_data[dim_index]->begin(), device_data[dim_index]->end(), aligned_device_data_ptr->begin() + dim_index * device_data[0]->size());
    }

    // ToDo: Free Memory
    return aligned_device_data_ptr;
}

// Reverts align_device_data
std::vector< dev_vec*> reorder_device_data(dev_vec* aligned_device_data, const uint8_t dim)
{
    std::vector< dev_vec* > device_data(dim);
    for(auto dim_index = 0; dim_index < dim; dim_index++)
    {
        // Generate device vector of reference vertices for each vector
        //device_data[dim_index] = &((*aligned_device_data)[dim_index * device_data[0]->size()]);
        device_data[dim_index] = new dev_vec(aligned_device_data->size()/dim);
        thrust::copy(
                aligned_device_data->begin() + dim_index * device_data[0]->size(),
                aligned_device_data->begin() + (dim_index + 1) * device_data[0]->size(),
                device_data[dim_index]->begin()
        );
    }
    // ToDo: Free Memory <-> is it possible to let the std::vector< dev_vec* > pointer point to the corresponding places of aligned_device_data??
    // This would enable that std::vector<dev_vec* > is only used as a wrapper for an in princinple aligned_vertices dev_vec -> this could even be procuded on stack! <-> use iterators instead??
    return device_data;
}

DevDatC cluster_device_data(
        uint maximum_expected_number_of_clusters,
        double upper_bound_for_min_distance,
        const DevDatC device_data,
        uint maximum_number_of_iterations)
{
    auto transposed_data = device_data.transpose_device_data();
    auto maximum_k = std::min(maximum_expected_number_of_clusters, uint(transposed_data.size()));

    // Perform clustering for different values of k
    std::vector<double> averaged_distances(maximum_k);
    for(auto k = 1; k <= maximum_k; k++)
    {
        KMeans kmeans(k, transposed_data);
        kmeans.apply(maximum_number_of_iterations);
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
        if((averaged_distances[i] < upper_bound_for_min_distance and appropriate_k == 1) or // first time below the threshold
           (averaged_distances[i] < upper_bound_for_min_distance and std::abs(adjacent_differences[i]) >  maximum_diff))
        {
            appropriate_k = i + 1;
            maximum_diff = std::abs(adjacent_differences[i]);
        }
    }

    /* if(adjacent_differences.size() > 1)
    {
        auto argmax_it = std::max_element(adjacent_differences.begin() + 1, adjacent_differences.end(), abs_compare(upper_bound_for_min_distance, averaged_distances));
        std::cout << "\nArgmax points to" << *argmax_it << std::endl;
        appropriate_k = argmax_it - adjacent_differences.begin() + 1;
    }
    else
        appropriate_k = 1;
    std::cout << "Appropriate k " << appropriate_k << std::endl;*/

    // ToDo: Plot result

    // Perform clustering ones more to get the cluster centers
    KMeans kmeans(appropriate_k, transposed_data);
    kmeans.apply(maximum_number_of_iterations);
    std::cout << "Final averaged distance: " << kmeans.get_mean_distance_to_cluster_centers() << " for k=" << appropriate_k << std::endl;
    std::vector < std::vector<double> > cluster_centers = kmeans.get_centers();
    return DevDatC(cluster_centers);
}