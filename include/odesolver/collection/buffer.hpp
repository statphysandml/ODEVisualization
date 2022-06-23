#ifndef PROJECT_BUFFER_HPP
#define PROJECT_BUFFER_HPP

#include <algorithm>
#include <tuple>

#include <devdat/header.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/collection/collection.hpp>


namespace odesolver {
    namespace collections {
        /** @brief Class managing a vector of collections allowing for a
        * step-wise computation of a recursive tree on the GPU */
        class Buffer
        {
        public:
            Buffer() = default;

            Buffer(std::shared_ptr<Collection> collection);

            std::tuple<std::vector<Collection*>, int, int> pop_collection_package(const int number_of_elements);

            void append_collection(const int internal_start_index, int internal_end_index, const std::vector<int> parent_indices);

            size_t len() const;

            static void get_collections_info(const std::vector<Collection*> &collections);

        private:
            std::vector<std::shared_ptr<Collection>> collections_;
            int initial_number_of_collections_in_depth_zero_;

            int collection_package_size_;
        };
    }
}

#endif //PROJECT_BUFFER_HPP
