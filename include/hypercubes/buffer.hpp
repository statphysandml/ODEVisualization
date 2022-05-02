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

    Buffer(std::shared_ptr<Node> node) : initial_number_of_nodes_in_depth_zero_(node->get_n_cubes())
    {
        nodes_.push_back(std::move(node));
        node_package_size_ = 0;
    }

    std::tuple<std::vector<Node*>, int, int> pop_node_package(const int number_of_cubes);

    void append_node(const int internal_start_index, int internal_end_index, const std::vector< int > parent_cube_indices)
    {
        // append new_node to node
        nodes_.push_back(std::make_shared<Node>(internal_start_index, internal_end_index, parent_cube_indices));
    }

    size_t len() const {
        return nodes_.size();
    }

    static void get_nodes_info(const std::vector<Node*> &nodes)
    {
        std::for_each(nodes.begin(), nodes.end(), [](Node* const& node) { node->info(); });
    }

private:
    std::vector<std::shared_ptr<Node>> nodes_;
    int initial_number_of_nodes_in_depth_zero_;

    int node_package_size_;
};

#endif //PROJECT_BUFFER_HPP
