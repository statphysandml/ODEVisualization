//
// Created by lukas on 13.03.19.
//

#ifndef PROJECT_BUFFER_HPP
#define PROJECT_BUFFER_HPP

#include <algorithm>
#include <tuple>

#include "../util/header.hpp"
#include "../util/monitor.hpp"
#include "node.hpp"


class Buffer
{
public:
    Buffer() = default;

    Buffer(Node* node) : initial_number_of_nodes_in_depth_zero(node->get_n_cubes())
    {
        nodes.push_back(node);
    }

    std::tuple< std::vector<Node* >, int, int > get_first_nodes(const int number_of_cubes);

    void add_nodes(std::vector<Node* > new_nodes)
    {
        // append new_nodes to nodes
        nodes.insert(nodes.end(), new_nodes.begin(), new_nodes.end());
    }

    size_t len() const {
        return nodes.size();
    }

    static void get_nodes_info(const std::vector<Node* > &nodes_)
    {
        std::for_each(nodes_.begin(), nodes_.end(), [](Node* const& node) { node->info(); });
    }

private:
    std::vector<Node* > nodes;
    int initial_number_of_nodes_in_depth_zero;
};

#endif //PROJECT_BUFFER_HPP
