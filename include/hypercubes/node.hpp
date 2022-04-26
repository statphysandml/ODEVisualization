//
// Created by lukas on 13.03.19.
//

#ifndef PROJECT_NODE_HPP
#define PROJECT_NODE_HPP

#include <numeric>
#include <iostream>

#include "nodecounter.hpp"


int compute_internal_end_index(const std::vector<int> &n_branches);


class Node : NodeCounter<Node>
{
public:
    Node(const int internal_start_index_, int internal_end_index_, const std::vector< int > parent_cube_indices_) :
    NodeCounter(parent_cube_indices_.size()),
    level_node_index(NodeCounter<Node>::objects_created[parent_cube_indices_.size()]),
    depth(parent_cube_indices_.size()),
    internal_start_index(internal_start_index_),
    internal_end_index(internal_end_index_), parent_cube_indices(parent_cube_indices_)
    {}

    // Generate a new node and adapt the end index of the current node
    Node* cut_node(const int cut_index)
    {
        Node * new_node_ptr = new Node(internal_start_index + cut_index + 1, internal_end_index, parent_cube_indices); // internal_start_index has been added recently
        internal_end_index = internal_start_index + cut_index + 1;
        return new_node_ptr;
    }

    int get_depth() const {
        return depth;
    }

    int get_internal_start_index() const {
        return internal_start_index;
    }

    int get_internal_end_index() const {
        return internal_end_index;
    }

    const std::vector<int>& get_parent_cube_indices() const {
        return parent_cube_indices;
    }

    int get_n_cubes() const {
        return internal_end_index - internal_start_index;
    }

    void info() const {
        std::cout << "\n\tLevel node index: " << level_node_index << std::endl;
        std::cout << "\tInternal start index: " << internal_start_index << std::endl;
        std::cout << "\tInternal end index: " << internal_end_index << std::endl;
        std::cout << "\tNumber of cubes: " << get_n_cubes() << std::endl;
        std::cout << "\tDepth: " << depth << std::endl;
        std::cout << "\tParent cube indices:";
        for(const auto& parent_cube_index: parent_cube_indices)
            std::cout << " " << parent_cube_index;
        std::cout << std::endl;
    }

    // if max depth -> node is undividable??

private:
    const int level_node_index; // corresponds to the i-th generated node in that depth
    const int depth;
    const std::vector< int > parent_cube_indices; // root, first node, second node, etc.

    const int internal_start_index; // inclusive
    int internal_end_index; // exclusive
};

#endif //PROJECT_NODE_HPP
