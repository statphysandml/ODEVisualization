#include <odesolver/collection/leaf.hpp>


namespace odesolver {
    namespace collections {
        Leaf::Leaf(const std::vector<int> indices) :
            indices_(indices), depth_(indices.size() - 1)
        {}

        void Leaf::info() const {
            std::cout << "\nSolution: " << std::endl;
            std::cout << "\tDepth: " << depth_ << std::endl;
            std::cout << "\tIndices:";
            for(const auto& index: indices_)
                std::cout << " " << index;
            std::cout << std::endl;
        }

        int Leaf::get_ith_depth_index(int i) const
        {
            return indices_[i];
        }

        json Leaf::to_json() const
        {
            return json {{"depth", depth_},
                         {"indices", indices_}};
        }

    }
}
