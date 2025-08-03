#ifndef PROJECT_COLLECTION_HPP
#define PROJECT_COLLECTION_HPP

#include <iostream>
#include <numeric>
#include <memory>

#include <odesolver/collection/counter.hpp>


namespace odesolver {
    namespace collections {
        int compute_internal_end_index(const std::vector<int> &n_branches);


        /** @brief Class for tracking and generating collections describing
         * packages of elements for later computations. The packages refer to
         * elements on the recursive search tree of the fixed point search. */
        class Collection : Counter<Collection>
        {
        public:
            Collection(const int internal_start_index, int internal_end_index, const std::vector<int> parent_indices=std::vector<int>{});

            ~Collection();

            // Generate a new collection and adapt the end index of the current collection
            std::shared_ptr<Collection> cut_collection(const int cut_index);

            int get_depth() const;

            int get_internal_start_index() const;

            int get_internal_end_index() const;

            const std::vector<int>& get_parent_indices() const;

            int size() const;

            void info() const;

        private:
            const int level_collection_index_; // corresponds to the i-th generated collection in that depth
            const int depth_;
            const std::vector<int> parent_indices_; // root, first collection, second collection, etc.

            const int internal_start_index_; // inclusive
            int internal_end_index_; // exclusive
        };
    }
}

#endif //PROJECT_COLLECTION_HPP
