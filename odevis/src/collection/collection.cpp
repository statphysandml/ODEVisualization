#include <odesolver/collection/collection.hpp>


namespace odesolver {
    namespace collections {
        int compute_internal_end_index(const std::vector<int> &n_branches)
        {
            return std::accumulate(n_branches.begin(), n_branches.end(), 1, std::multiplies<int>());
        }


        Collection::Collection(const int internal_start_index, int internal_end_index, const std::vector<int> parent_indices) :
        Counter(parent_indices.size()),
        level_collection_index_(Counter<Collection>::objects_created_[parent_indices.size()]),
        depth_(parent_indices.size()),
        internal_start_index_(internal_start_index),
        internal_end_index_(internal_end_index), parent_indices_(parent_indices)
        {}

        Collection::~Collection()
        {
            --Counter::objects_alive_[get_depth()];
        }

        // Generate a new collection and adapt the end index of the current collection
        std::shared_ptr<Collection> Collection::cut_collection(const int cut_index)
        {
            std::shared_ptr<Collection> new_collection_ptr = std::make_shared<Collection>(internal_start_index_ + cut_index + 1, internal_end_index_, parent_indices_);
            internal_end_index_ = internal_start_index_ + cut_index + 1;
            return std::move(new_collection_ptr);
        }

        int Collection::get_depth() const {
            return depth_;
        }

        int Collection::get_internal_start_index() const {
            return internal_start_index_;
        }

        int Collection::get_internal_end_index() const {
            return internal_end_index_;
        }

        const std::vector<int>& Collection::get_parent_indices() const {
            return parent_indices_;
        }

        int Collection::size() const {
            return internal_end_index_ - internal_start_index_;
        }

        void Collection::info() const {
            std::cout << "\n\tLevel collection index: " << level_collection_index_ << std::endl;
            std::cout << "\tInternal start index: " << internal_start_index_ << std::endl;
            std::cout << "\tInternal end index: " << internal_end_index_ << std::endl;
            std::cout << "\tNumber of elements: " << size() << std::endl;
            std::cout << "\tDepth: " << depth_ << std::endl;
            std::cout << "\tParent indices:";
            for(const auto& parent_index: parent_indices_)
                std::cout << " " << parent_index;
            std::cout << std::endl;
        }
    }
}
