#ifndef PROJECT_LEAF_HPP
#define PROJECT_LEAF_HPP

#include <iostream>

#include <param_helper/json.hpp>

using json = nlohmann::json;


namespace odesolver {
    namespace collections {
        /** @brief Encoding of elements at the end of the recursive search tree. The indices allow for a computation of the vertices of the respective elements based on the knowledge of the different search ranges and number of branches per dimension, for example. */
        class Leaf
        {
        public:
            Leaf(const std::vector<int> indices);

            void info() const;

            int get_ith_depth_index(int i) const;

            json to_json() const;

        private:
            const int depth_;
            const std::vector<int> indices_; // root, first element, second element, etc.
        };
    }
}

#endif //PROJECT_LEAF_HPP
