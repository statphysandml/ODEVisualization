//
// Created by lukas on 06.04.19.
//

#ifndef PROJECT_NODESEXPANDER_HPP
#define PROJECT_NODESEXPANDER_HPP

#include "../util/header.hpp"
#include "../util/dev_dat.hpp"
#include "../util/monitor.hpp"
#include "node.hpp"
#include "gridcomputationwrapper.hpp"
#include "../extern/thrust_functors.hpp"

/* Class for extracting the information about parent cube indices and depth per cube for the given nodes
 * -> as a result one obtains vectors of parent cube indices and a vector of the depth per cube */


class NodesExpander
{
public:
    NodesExpander(const int total_number_of_cubes_, const int maximum_depth_,
               const std::vector<Node* > &nodes_) :
               total_number_of_cubes(total_number_of_cubes_), maximum_depth(maximum_depth_),
               number_of_nodes(nodes_.size()), nodes(nodes_),
               collected_parent_cube_indices(thrust::host_vector< dev_vec_int >(maximum_depth)), // One vector for each depths that contains the cube indices of the corresponding depth and node
               number_of_cubes_per_node(dev_vec_int(number_of_nodes)), // Contains number of cubes for each node
               depth_per_node(dev_vec_int(number_of_nodes)) // Contains final depth for each node
    {}

    // Fills based on the nodes the given arrays for a later computation of vertices
    /* Example output
     * Number of cubes per node: 4 8 8 // number_of_cubes_per_node
     * Depth per node: 5 5 5 // depth_per_node
     * Collected parent cube indices in depth 0: 90 90 90 // ...collected_parent_cube_indices...
     * Collected parent cube indices in depth 1: 3 3 3
     * Collected parent cube indices in depth 2: 1 1 1
     * Collected parent cube indices in depth 3: 1 1 1
     * Collected parent cube indices in depth 4: 3 5 7
     */
    void extract_node_information();

    // Cuda code - Expands the node information according to the given number of cubes per node with respect to
    // the common parent cube indices -> fills the expanded vectors
    /* Example output
     * Expanded cube indices after filling with individual cube indices in depth 0: 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90
     * Expanded cube indices after filling with individual cube indices in depth 1: 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
     * Expanded cube indices after filling with individual cube indices in depth 2: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     * Expanded cube indices after filling with individual cube indices in depth 3: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     * Expanded cube indices after filling with individual cube indices in depth 4: 3 3 3 3 5 5 5 5 5 5 5 5 7 7 7 7 7 7 7 7
     * Expanded cube indices after filling with individual cube indices in depth 5: 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
     * Expanded depth per node: 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
     */
    GridComputationWrapper expand_node_information_according_to_number_of_nodes();

private:
    const size_t number_of_nodes;
    const std::vector<Node* > &nodes;

    const int total_number_of_cubes;
    const int maximum_depth;

    // Initialize vectors that contain information of each node for a latter expansion and reconstruction
    // of vertex indices
    thrust::host_vector< dev_vec_int > collected_parent_cube_indices; // One vector for each depths that contains the cube indices of the corresponding depth and node
    dev_vec_int number_of_cubes_per_node; // Contains number of cubes for each node
    dev_vec_int depth_per_node; // Contains final depth for each node
};

#endif //PROJECT_NODESEXPANDER_HPP
