//
// Created by lukas on 13.03.19.
//

#ifndef PROJECT_BUFFER_HPP
#define PROJECT_BUFFER_HPP

#include <algorithm>
#include <tuple>

#include "../odesolver/util/header.hpp"
#include "../odesolver/util/monitor.hpp"
#include "node.hpp"


class Buffer
{
public:
    Buffer() = default;

    Buffer(Node* node) : initial_number_of_nodes_in_depth_zero(node->get_n_cubes())
    {
        nodes.push_back(node);
    }

    std::tuple<std::vector<Node*>, int, int> get_first_nodes(const int number_of_cubes);

    void append_node(const int internal_start_index, int internal_end_index, const std::vector< int > parent_cube_indices)
    {
        // append new_nodes to nodes
        nodes.push_back(new Node(internal_start_index, internal_end_index, parent_cube_indices));
    }

    size_t len() const {
        return nodes.size();
    }

    static void get_nodes_info(const std::vector<Node*> &nodes_)
    {
        std::for_each(nodes_.begin(), nodes_.end(), [](Node* const& node) { node->info(); });
    }

private:
    std::vector<Node*> nodes;
    int initial_number_of_nodes_in_depth_zero;
};

#endif //PROJECT_BUFFER_HPP
