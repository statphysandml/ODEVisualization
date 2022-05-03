//
// Created by lukas on 13.03.19.
//

#ifndef PROJECT_NODE_HPP
#define PROJECT_NODE_HPP

#include <iostream>
#include <numeric>
#include <memory>

#include "nodecounter.hpp"


int compute_internal_end_index(const std::vector<int> &n_branches);


/** @brief Class for tracking and generating nodes describing packages of
 * hypercubes for later computations. The packages refer to hypercubes on the
 * recursive search tree of the fixed point search. */
class Node : NodeCounter<Node>
{
public:
    Node(const int internal_start_index, int internal_end_index, const std::vector< int > parent_cube_indices) :
    NodeCounter(parent_cube_indices.size()),
    level_node_index_(NodeCounter<Node>::objects_created_[parent_cube_indices.size()]),
    depth_(parent_cube_indices.size()),
    internal_start_index_(internal_start_index),
    internal_end_index_(internal_end_index), parent_cube_indices_(parent_cube_indices)
    {}

    ~Node()
    {
        --NodeCounter::objects_alive_[get_depth()];
    }

    // Generate a new node and adapt the end index of the current node
    std::shared_ptr<Node> cut_node(const int cut_index)
    {
        std::shared_ptr<Node> new_node_ptr = std::make_shared<Node>(internal_start_index_ + cut_index + 1, internal_end_index_, parent_cube_indices_);
        internal_end_index_ = internal_start_index_ + cut_index + 1;
        return std::move(new_node_ptr);
    }

    int get_depth() const {
        return depth_;
    }

    int get_internal_start_index() const {
        return internal_start_index_;
    }

    int get_internal_end_index() const {
        return internal_end_index_;
    }

    const std::vector<int>& get_parent_cube_indices() const {
        return parent_cube_indices_;
    }

    int get_n_cubes() const {
        return internal_end_index_ - internal_start_index_;
    }

    void info() const {
        std::cout << "\n\tLevel node index: " << level_node_index_ << std::endl;
        std::cout << "\tInternal start index: " << internal_start_index_ << std::endl;
        std::cout << "\tInternal end index: " << internal_end_index_ << std::endl;
        std::cout << "\tNumber of cubes: " << get_n_cubes() << std::endl;
        std::cout << "\tDepth: " << depth_ << std::endl;
        std::cout << "\tParent cube indices:";
        for(const auto& parent_cube_index: parent_cube_indices_)
            std::cout << " " << parent_cube_index;
        std::cout << std::endl;
    }

private:
    const int level_node_index_; // corresponds to the i-th generated node in that depth
    const int depth_;
    const std::vector< int > parent_cube_indices_; // root, first node, second node, etc.

    const int internal_start_index_; // inclusive
    int internal_end_index_; // exclusive
};

#endif //PROJECT_NODE_HPP
