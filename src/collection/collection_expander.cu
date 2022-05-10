#include <odesolver/collection/collection_expander.hpp>


namespace odesolver {
    namespace collections {
        CollectionExpander::CollectionExpander(
            const int maximum_depth,
            const int number_of_collections
        ):
            collected_parent_indices_(thrust::host_vector<dev_vec_int>(maximum_depth)),
            number_of_elements_per_collection_(dev_vec_int(number_of_collections)),
            depth_per_collection_(dev_vec_int(number_of_collections))
        {
            // Initialize collected parent indices
            auto n_collections = number_of_collections; // to avoid passing "this" within the lambda capture
            thrust::generate(collected_parent_indices_.begin(), collected_parent_indices_.end(), [n_collections]() { return dev_vec_int(n_collections, 0); });

            expected_number_of_elements_ = 0;
            expected_depth_ = 0;
        }


        void CollectionExpander::extract_collection_information(const std::vector<Collection*> &collection_package)
        {
            // Collect counts and values for the expansion
            expected_depth_ = 0;

            for(auto collection_index = 0; collection_index < collection_package.size(); collection_index++)
            {
                // Collect indices for each dimension
                const std::vector<int> &parent_indices = collection_package[collection_index]->get_parent_indices();
                // print_range("Parent indices", parent_indices.begin(), parent_indices.end());
                int depth = collection_package[collection_index]->get_depth();
                for(auto depth_index = 0; depth_index < depth; depth_index++)
                    collected_parent_indices_[depth_index][collection_index] = parent_indices[depth_index];

                // Derive depths
                depth_per_collection_[collection_index] = depth;
                if(depth > expected_depth_)
                    expected_depth_ = depth;

                // Counter for latter expansion
                number_of_elements_per_collection_[collection_index] = collection_package[collection_index]->size();
            }

            expected_number_of_elements_ = thrust::reduce(number_of_elements_per_collection_.begin(), number_of_elements_per_collection_.end());

            // Testing
            /*auto i = 0;
            print_range("Number of elements per collection", number_of_elements_per_collection_.begin(), number_of_elements_per_collection_.end());
            print_range("Depth per collection", depth_per_collection_.begin(), depth_per_collection_.end());
            for(auto &elem : collected_parent_indices_)
            {
                print_range("Collected parent indices in depth " + std::to_string(i), elem.begin(), elem.end());
                i++;
            }*/

            /* Example output
            * Number of elementss per collection: 4 8 8 // number_of_elements_per_collection
            * Depth per collection: 5 5 5 // depth_per_collection
            * Collected parent indices in depth 0: 90 90 90 // ...collected_parent_indices...
            * Collected parent indices in depth 1: 3 3 3
            * Collected parent indices in depth 2: 1 1 1
            * Collected parent indices in depth 3: 1 1 1
            * Collected parent indices in depth 4: 3 5 7
            */
        }


        void CollectionExpander::expand_collection_information(
            const std::vector<Collection*> &collection_package,
            odesolver::DevDatInt& expanded_element_indices,
            odesolver::DimensionIteratorInt& expanded_depth_per_element
        )
        {
            if(expected_number_of_elements_ > expanded_element_indices.n_elems())
            {
                std::cerr << "Number of elements in expanded_element_indices is too small in comparison to the expected number of elements." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            if(expected_depth_ + 1 > expanded_element_indices.dim_size())
            {
                std::cerr << "Provided maximum depth in expanded_element_indices is too small in comparison to the expected maximum recursive depth." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // Expand parent indices
            for(auto depth_index = 0; depth_index < expected_depth_; depth_index++) {
                expand(number_of_elements_per_collection_.begin(), number_of_elements_per_collection_.end(), collected_parent_indices_[depth_index].begin(), expanded_element_indices[depth_index].begin());
            }

            // Expand depth
            expand(number_of_elements_per_collection_.begin(), number_of_elements_per_collection_.end(), depth_per_collection_.begin(), expanded_depth_per_element.begin());

            // [ Fill expanded indices with individual indices - fills last row of expanded indices
            
            // Generate iterators for each depth
            dev_iter_vec_int depth_iterator(expected_depth_ + 1);
            for(auto depth_index = 0; depth_index < expected_depth_ + 1; depth_index++)
                depth_iterator[depth_index] = new dev_iterator_int(expanded_element_indices[depth_index].begin());

            // Fill remaining indices
            auto collection_index = 0;
            for(auto &collection : collection_package) {
                int depth = depth_per_collection_[collection_index]; //Equivalent to collection->get_depth();
                // Extract individual index and copy into correct depth
                *depth_iterator[depth] = thrust::copy(thrust::make_counting_iterator(collection->get_internal_start_index()),
                                                    thrust::make_counting_iterator(collection->get_internal_end_index()),
                                                    *depth_iterator[depth]);
                // Increment all other depth iterators by the amount of vertices according to the number of elements in the considered collection
                for (auto depth_index = 0; depth_index < expected_depth_ + 1; depth_index++) {
                    if (depth_index != depth)
                        *depth_iterator[depth_index] = *depth_iterator[depth_index] + number_of_elements_per_collection_[collection_index]; // Equivalent to collection->size();
                }
                collection_index++;
            }

            for(auto depth_index = 0; depth_index < expected_depth_ + 1; depth_index++)
                delete depth_iterator[depth_index];
            // ]

            /* Example output
            * Expanded indices after filling with individual indices in depth 0: 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90
            * Expanded indices after filling with individual indices in depth 1: 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
            * Expanded indices after filling with individual indices in depth 2: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
            * Expanded indices after filling with individual indices in depth 3: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
            * Expanded indices after filling with individual indices in depth 4: 3 3 3 3 5 5 5 5 5 5 5 5 7 7 7 7 7 7 7 7
            * Expanded indices after filling with individual indices in depth 5: 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
            * Expanded depth per collection: 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
            */
        }
    }
}
