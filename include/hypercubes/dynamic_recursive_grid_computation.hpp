#ifndef PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP
#define PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP

#include "../odesolver/util/dev_dat.hpp"
#include "../odesolver/util/computation_parameters.hpp"
#include "../flow_equation_interface/flow_equation.hpp"
#include "buffer.hpp"
#include "hypercubes.hpp"
#include "node.hpp"
#include "nodesexpander.hpp"


class DynamicRecursiveGridComputation
{
public:
    // Possible modes -> correspond to different possible usages of hypercubes
    enum VertexMode { CenterVertices, CubeVertices, ReferenceVertices};

    DynamicRecursiveGridComputation(
        const int number_of_cubes_per_gpu_call=20000,
        const int maximum_number_of_gpu_calls=1000
    );

    void initialize(
        const std::vector<std::vector<int>> n_branches_per_depth,
        const std::vector<std::pair<cudaT, cudaT>> lambda_ranges,
        VertexMode vertex_mode
    );

    virtual void next(odesolver::DevDatC &vertices);

    bool finished();

    bool check_status();

    const std::vector<Node*>& get_node_package();

    Buffer& get_buffer();

    HyperCubes& get_hypercubes();

protected:
    int number_of_cubes_per_gpu_call_;
    int maximum_number_of_gpu_calls_;

    VertexMode vertex_mode_;
    HyperCubes hypercubes_;
    Buffer buffer_;
    int c_;
    
    std::vector<Node*> node_package_;
    int expected_number_of_cubes_;
    int expected_maximum_depth_;
};

#endif //PROGRAM_DYNAMIC_RECURSIVE_GRID_COMPUTATION_HPP
