#include "../../include/hypercubes/buffer.hpp"

std::tuple<std::vector<Node*>, int, int> Buffer::pop_node_package(const int number_of_cubes)
{
    int total_number_of_cubes = 0;
    int maximum_depth = 0;

    // Erase already processed nodes and initialize node_iterator
    nodes_.erase(nodes_.begin(), nodes_.begin() + node_package_size_);
    auto node_iterator = nodes_.begin();

    NodeCounter<Node>::number_of_alive_nodes_per_depth();
    std::cout << std::endl;

    if(monitor) {
        std::cout << "\nCollect nodes for the gpu" << std::endl;
        std::cout << "\tCurrent number of nodes in buffer: " << len() << std::endl;
    }

    // Iterate over nodes and sum up the total number of cubes
    while(total_number_of_cubes < number_of_cubes and node_iterator != nodes_.end())
    {
        total_number_of_cubes += (*node_iterator)->get_n_cubes();
        maximum_depth = std::max(maximum_depth, (*node_iterator)->get_depth());
        if(monitor)
            std::cout << "\tNumber of cubes within this node: " << (*node_iterator)->get_n_cubes() << std::endl;
        if((*node_iterator)->get_depth() == 0)
            std::cout << "\tRemaining number of cubes in depth 0: " << (*node_iterator)->get_n_cubes() << " = " << (*node_iterator)->get_n_cubes()*100.0/initial_number_of_nodes_in_depth_zero_ << "%" << std::endl;
        node_iterator++;
    }

    // Divide last node into two nodes so that the total_number_of_cubes doesn't exceed the desired number_of_cubes
    if(total_number_of_cubes > number_of_cubes)
    {
        // Iterate back
        node_iterator--;
        // Divide node on current iterator position
        const int overhead_number_of_cubes = total_number_of_cubes - number_of_cubes;
        std::shared_ptr<Node> new_node_ptr = (*node_iterator)->cut_node((*node_iterator)->get_n_cubes() - overhead_number_of_cubes - 1);
        // Include new node into nodes
        node_iterator++;
        node_iterator = nodes_.insert(node_iterator, std::move(new_node_ptr)); // node_iterator points to the inserted node
        total_number_of_cubes = number_of_cubes;
    }


    // return first elements till iterator and delete the nodes in the buffer list
    std::vector<Node*> node_package;
    std::transform(nodes_.begin(), node_iterator, std::back_inserter(node_package), [](const std::shared_ptr<Node>& node_ptr) {
        return node_ptr.get();
    });
    // std::vector<Node*> node_package(nodes_.begin(), node_iterator);

    // std::cout << "\n### To be returned nodes" << std::endl;
    // get_nodes_info(node_package);
    //std::cout << "\n### Nodes after erase" << std::endl;
    //get_nodes_info(nodes);

    node_package_size_ = node_iterator - nodes_.begin();

    return std::make_tuple(node_package, total_number_of_cubes, maximum_depth);
}