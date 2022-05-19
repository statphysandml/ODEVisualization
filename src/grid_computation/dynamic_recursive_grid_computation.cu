#include <odesolver/grid_computation/dynamic_recursive_grid_computation.hpp>


namespace odesolver {
    namespace gridcomputation {
        DynamicRecursiveGridComputation::DynamicRecursiveGridComputation(
            const int number_of_cubes_per_gpu_call,
            const int maximum_number_of_gpu_calls) :
            number_of_cubes_per_gpu_call_(number_of_cubes_per_gpu_call),
            maximum_number_of_gpu_calls_(maximum_number_of_gpu_calls)
        {}

        void DynamicRecursiveGridComputation::initialize(const std::vector<std::vector<int>> n_branches_per_depth,
            const std::vector<std::pair<cudaT, cudaT>> variable_ranges, VertexMode vertex_mode)
        {
            vertex_mode_ = vertex_mode;

            hypercubes_ = GridComputation(n_branches_per_depth, variable_ranges);

            std::shared_ptr<odesolver::collections::Collection> root_collection_ptr = std::make_shared<odesolver::collections::Collection>(0, odesolver::collections::compute_internal_end_index(hypercubes_.n_branches_per_depth()[0]), std::vector<int>{});
            buffer_ = odesolver::collections::Buffer(std::move(root_collection_ptr));

            c_ = 0;
        }

        void DynamicRecursiveGridComputation::next(odesolver::DevDatC &vertices)
        {
            // Get collections for the gpu from buffer
            std::tie(collection_package_, expected_number_of_cubes_, expected_maximum_depth_) = buffer_.pop_collection_package(number_of_cubes_per_gpu_call_);

            if(monitor) {
                std::cout << "\n### odesolver::collections::Collections for the qpu: " << collection_package_.size() << ", total number of cubes: "
                        << expected_number_of_cubes_ << std::endl;
                buffer_.get_collections_info(collection_package_);
            }

            // Expand collections
            GridComputationWrapper grid_computation_wrapper(expected_number_of_cubes_, expected_maximum_depth_);
            
            odesolver::collections::CollectionExpander collectionsexpander(expected_maximum_depth_, collection_package_.size());

            // Fill vectors of length number of collections
            collectionsexpander.extract_collection_information(collection_package_);
            
            // Fill vectors of length total number of cubes
            collectionsexpander.expand_collection_information(
                collection_package_,
                grid_computation_wrapper.expanded_element_indices_,
                grid_computation_wrapper.expanded_depth_per_element_
            );

            // Testing
            if(monitor)
                grid_computation_wrapper.print_expanded_vectors();

            // Compute vertices
            if(vertex_mode_ == CenterVertices)
            {
                vertices = odesolver::DevDatC(hypercubes_.dim() ,expected_number_of_cubes_);
                hypercubes_.compute_reference_vertices(vertices, grid_computation_wrapper);
            }
            else if(vertex_mode_ == CubeVertices)
            {
                vertices = odesolver::DevDatC(hypercubes_.dim(), expected_number_of_cubes_ * pow(2, hypercubes_.dim()));
                hypercubes_.compute_cube_vertices(vertices, grid_computation_wrapper);
            }
            else
            {
                vertices = odesolver::DevDatC(hypercubes_.dim(),expected_number_of_cubes_);
                hypercubes_.compute_cube_center_vertices(vertices, grid_computation_wrapper);
            }

            c_++;
        }

        bool DynamicRecursiveGridComputation::finished()
        {
            return (buffer_.len() == 0) or (c_ > maximum_number_of_gpu_calls_);
        }

        bool DynamicRecursiveGridComputation::check_status()
        {
            if(c_ > maximum_number_of_gpu_calls_)
            {
                std::cout << "Finished because the maximum number of gpu calls has been exceeded." << std::endl;
                return false;
            }
            else
                return true;
        }

        const std::vector<odesolver::collections::Collection*>& DynamicRecursiveGridComputation::get_collection_package()
        {
            return collection_package_;
        }

        odesolver::collections::Buffer& DynamicRecursiveGridComputation::get_buffer()
        {
            return buffer_;
        }

        GridComputation& DynamicRecursiveGridComputation::get_hypercubes()
        {
            return hypercubes_;
        }
    }
}
