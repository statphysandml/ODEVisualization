//
// Created by lukas on 13.03.19.
//

#ifndef PROJECT_LEAF_HPP
#define PROJECT_LEAF_HPP

#include "param_helper/json.hpp"

using json = nlohmann::json;

class Leaf
{
public:
    Leaf(const std::vector< int > cube_indices_) :
            cube_indices(cube_indices_), depth(cube_indices_.size() - 1)
    {}

    void info() const {
        std::cout << "\nSolution: " << std::endl;
        std::cout << "\tDepth: " << depth << std::endl;
        std::cout << "\tCube indices:";
        for(const auto& cube_index: cube_indices)
            std::cout << " " << cube_index;
        std::cout << std::endl;
    }

    int get_ith_cube_depth_index(int i) const
    {
        return cube_indices[i];
    }

    json to_json() const
    {
        return json {{"depth", depth},
                     {"cube_indices", cube_indices}};
    }

private:
    const int depth;
    const std::vector< int > cube_indices; // root, first cube, second cube, usw.
};

#endif //PROJECT_LEAF_HPP
