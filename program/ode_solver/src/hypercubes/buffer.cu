#include "../../include/hypercubes/buffer.hpp"

std::tuple< std::vector<Node* >, int, int > Buffer::get_first_nodes(const int number_of_cubes)
{
    int total_number_of_cubes = 0;
    int maximum_depth = 0;
    auto node_iterator = nodes.begin();

    if(monitor) {
        std::cout << "\nStart to get nodes for the gpu" << std::endl;
        std::cout << "\tCurrent number of nodes in buffer: " << len() << std::endl;
    }

    // Iterate over nodes and sum up the total number of cubes
    while(total_number_of_cubes < number_of_cubes and node_iterator != nodes.end())
    {
        total_number_of_cubes += (*node_iterator)->get_n_cubes();
        maximum_depth = std::max(maximum_depth, (*node_iterator)->get_depth());
        if(monitor)
            std::cout << "\tNumber of cubes within this node: " << (*node_iterator)->get_n_cubes() << std::endl;
        if((*node_iterator)->get_depth() == 0)
            std::cout << "\tRemaining number of cubes in depth 0: " << (*node_iterator)->get_n_cubes() << " = " << (*node_iterator)->get_n_cubes()*100.0/initial_number_of_nodes_in_depth_zero << "%" << std::endl;
        node_iterator++;
    }

    if(total_number_of_cubes > number_of_cubes)
    {
        // Iterate back
        node_iterator--;
        // Divide node on current iterator position
        const int overhead_number_of_cubes = total_number_of_cubes - number_of_cubes;
        Node * new_node_ptr = (*node_iterator)->cut_node((*node_iterator)->get_n_cubes() - overhead_number_of_cubes - 1);
        // Include new node into nodes
        node_iterator++;
        node_iterator = nodes.insert(node_iterator, new_node_ptr); // node_iterator points to the inserted node
        total_number_of_cubes = number_of_cubes;
    }


    // return first elements till iterator and delete the nodes in the buffer list
    std::vector<Node* > to_be_returned_nodes(nodes.begin(), node_iterator);

    nodes.erase(nodes.begin(), node_iterator);

    // std::cout << "\n### To be returned nodes" << std::endl;
    // get_nodes_info(to_be_returned_nodes);
    //std::cout << "\n### Nodes after erase" << std::endl;
    //get_nodes_info(nodes);

    return std::make_tuple(to_be_returned_nodes, total_number_of_cubes, maximum_depth);
}