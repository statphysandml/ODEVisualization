//
// Created by lukas on 13.03.19.
//

#ifndef PROJECT_LEAF_HPP
#define PROJECT_LEAF_HPP

#include <param_helper/json.hpp>

using json = nlohmann::json;

/** @brief Encoding of hypercubes at the end of the recursive search tree. The cube indices allow for a computation of the vertices of the respective hypercubes based on the knowledge of the different search ranges and number of branches per dimension. */
class Leaf
{
public:
    Leaf(const std::vector<int> cube_indices) :
            cube_indices_(cube_indices), depth_(cube_indices.size() - 1)
    {}

    void info() const {
        std::cout << "\nSolution: " << std::endl;
        std::cout << "\tDepth: " << depth_ << std::endl;
        std::cout << "\tCube indices:";
        for(const auto& cube_index: cube_indices_)
            std::cout << " " << cube_index;
        std::cout << std::endl;
    }

    int get_ith_cube_depth_index(int i) const
    {
        return cube_indices_[i];
    }

    json to_json() const
    {
        return json {{"depth", depth_},
                     {"cube_indices", cube_indices_}};
    }

private:
    const int depth_;
    const std::vector<int> cube_indices_; // root, first cube, second cube, etc.
};

#endif //PROJECT_LEAF_HPP
