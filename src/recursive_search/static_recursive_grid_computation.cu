#include <odesolver/recursive_search/static_recursive_grid_computation.hpp>


namespace odesolver {
    namespace recursivesearch {
        StaticRecursiveGridComputation::StaticRecursiveGridComputation(
            const int maximum_recursion_depth,
            const int number_of_cubes_per_gpu_call,
            const int maximum_number_of_gpu_calls
        ) : DynamicRecursiveGridComputation(number_of_cubes_per_gpu_call, maximum_number_of_gpu_calls),
            grid_computation_wrapper_(odesolver::gridcomputation::GridComputationWrapper(
            number_of_cubes_per_gpu_call_, maximum_recursion_depth))
        {}

        void StaticRecursiveGridComputation::next(devdat::DevDatC &vertices)
        {
            // Get collections for the gpu from buffer
            std::tie(collection_package_, expected_number_of_cubes_, expected_maximum_depth_) = buffer_.pop_collection_package(number_of_cubes_per_gpu_call_);

            if(monitor) {
                std::cout << "\n### Collections for the qpu: " << collection_package_.size() << ", total number of cubes: "
                        << expected_number_of_cubes_ << std::endl;
                buffer_.get_collections_info(collection_package_);
            }
            
            odesolver::collections::CollectionExpander collectionsexpander(expected_maximum_depth_, collection_package_.size());

            // Fill vectors of length number of collections
            collectionsexpander.extract_collection_information(collection_package_);
            
            // Fill vectors of length total number of cubes
            grid_computation_wrapper_.expanded_element_indices_.set_N(expected_number_of_cubes_);
            grid_computation_wrapper_.expanded_depth_per_element_wrapper_.set_N(expected_number_of_cubes_);
            collectionsexpander.expand_collection_information(
                collection_package_,
                grid_computation_wrapper_.expanded_element_indices_,
                grid_computation_wrapper_.expanded_depth_per_element_
            );

            // Compute vertices
            if(vertex_mode_ == CenterVertices)
            {
                vertices.set_N(expected_number_of_cubes_);
                hypercubes_.compute_cube_center_vertices(vertices, grid_computation_wrapper_, expected_maximum_depth_);
            }
            else if(vertex_mode_ == CubeVertices)
            {
                vertices.set_N(expected_number_of_cubes_ * pow(2, hypercubes_.dim()));
                hypercubes_.compute_cube_vertices(vertices, grid_computation_wrapper_, expected_maximum_depth_);
            }
            else
            {
                vertices.set_N(expected_number_of_cubes_);
                hypercubes_.compute_reference_vertices(vertices, grid_computation_wrapper_, expected_maximum_depth_);
            }

            c_++;
        }
    }
}