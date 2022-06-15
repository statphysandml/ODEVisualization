#ifndef PROJECT_LEAF_HPP
#define PROJECT_LEAF_HPP

#include <iostream>

#include <nlohmann/json.hpp>
#include <param_helper/filesystem.hpp>

using json = nlohmann::json;


namespace odesolver {
    namespace collections {
        /** @brief Encoding of elements at the end of the recursive search tree. The indices allow for a computation of the vertices of the respective elements based on the knowledge of the different search ranges and number of branches per dimension, for example. */
        struct Leaf
        {
            Leaf(const std::vector<int> indices);

            void info() const;

            int get_ith_depth_index(int i) const;

            json to_json() const;

            const int depth_;
            const std::vector<int> indices_; // root, first element, second element, etc.
        };


        void write_leaves_to_file(std::string rel_dir, std::vector<odesolver::collections::Leaf*> leaves);

        std::vector<std::shared_ptr<odesolver::collections::Leaf>> load_leaves(std::string rel_dir);
    }
}

#endif //PROJECT_LEAF_HPP
