#include <odesolver/collection/buffer.hpp>


namespace odesolver {
    namespace collections {
        Buffer::Buffer(std::shared_ptr<Collection> collection) : initial_number_of_collections_in_depth_zero_(collection->size())
        {
            collections_.push_back(std::move(collection));
            collection_package_size_ = 0;
        }

        
        std::tuple<std::vector<Collection*>, int, int> Buffer::pop_collection_package(const int number_of_elements)
        {
            int total_number_of_elements = 0;
            int maximum_depth = 0;

            // Erase already processed collections and initialize collection_iterator
            collections_.erase(collections_.begin(), collections_.begin() + collection_package_size_);
            auto collection_iterator = collections_.begin();

            Counter<Collection>::number_of_alive_collections_per_depth();
            std::cout << std::endl;

            if(monitor) {
                std::cout << "\nCollect collections for the gpu" << std::endl;
                std::cout << "\tCurrent number of collections in buffer: " << len() << std::endl;
            }

            // Iterate over collections and sum up the total number of elements
            while(total_number_of_elements < number_of_elements and collection_iterator != collections_.end())
            {
                total_number_of_elements += (*collection_iterator)->size();
                maximum_depth = std::max(maximum_depth, (*collection_iterator)->get_depth());
                if(monitor)
                    std::cout << "\tNumber of elements within this collection: " << (*collection_iterator)->size() << std::endl;
                if((*collection_iterator)->get_depth() == 0)
                    std::cout << "\tRemaining number of elements in depth 0: " << (*collection_iterator)->size() << " = " << (*collection_iterator)->size()*100.0/initial_number_of_collections_in_depth_zero_ << "%" << std::endl;
                collection_iterator++;
            }

            // Divide last collection into two collections so that the total_number_of_elements doesn't exceed the desired number_of_elements
            if(total_number_of_elements > number_of_elements)
            {
                // Iterate back
                collection_iterator--;
                // Divide collection on current iterator position
                const int overhead_number_of_elements = total_number_of_elements - number_of_elements;
                std::shared_ptr<Collection> new_collection_ptr = (*collection_iterator)->cut_collection((*collection_iterator)->size() - overhead_number_of_elements - 1);
                // Include new collection into collections
                collection_iterator++;
                collection_iterator = collections_.insert(collection_iterator, std::move(new_collection_ptr)); // collection_iterator points to the inserted collection
                total_number_of_elements = number_of_elements;
            }


            // return first elements till iterator and delete the collections in the buffer list
            std::vector<Collection*> collection_package;
            std::transform(collections_.begin(), collection_iterator, std::back_inserter(collection_package), [](const std::shared_ptr<Collection>& collection_ptr) {
                return collection_ptr.get();
            });
            // std::vector<Collection*> collection_package(collections_.begin(), collection_iterator);

            // std::cout << "\n### To be returned collections" << std::endl;
            // get_collections_info(collection_package);
            //std::cout << "\n### Collections after erase" << std::endl;
            //get_collections_info(collections);

            collection_package_size_ = collection_iterator - collections_.begin();

            return std::make_tuple(collection_package, total_number_of_elements, maximum_depth + 1);
        }

        void Buffer::append_collection(const int internal_start_index, int internal_end_index, const std::vector<int> parent_indices)
        {
            // append new_collection to collection
            collections_.push_back(std::make_shared<Collection>(internal_start_index, internal_end_index, parent_indices));
        }

        size_t Buffer::len() const {
            return collections_.size();
        }

        void Buffer::get_collections_info(const std::vector<Collection*> &collections)
        {
            std::for_each(collections.begin(), collections.end(), [](Collection* const& collection) { collection->info(); });
        }
    }
}