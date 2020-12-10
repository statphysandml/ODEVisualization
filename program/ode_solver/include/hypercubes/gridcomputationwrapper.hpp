//
// Created by lukas on 23.09.19.
//

#ifndef PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP
#define PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP

#include "../util/dev_dat.hpp"

struct GridComputationWrapper
{
    /* GridComputationWrapper() :
            expanded_cube_indices(DevDatInt()),
            expanded_depth_per_cube_wrapper(DevDatInt(1, 1)),
            expanded_depth_per_cube(expanded_depth_per_cube_wrapper[0])
    {}*/

    GridComputationWrapper(const int total_number_of_cubes, const int maximum_depth, const cudaT init_depth_val = 0) :
            expanded_cube_indices(DevDatInt(maximum_depth, total_number_of_cubes)),
            expanded_depth_per_cube_wrapper(DevDatInt(1, total_number_of_cubes, init_depth_val)),
            expanded_depth_per_cube(expanded_depth_per_cube_wrapper[0])
    {}

    /* // Copy Constructor
    GridComputationWrapper(const GridComputationWrapper& b) :
            expanded_cube_indices(b.expanded_cube_indices),
            expanded_depth_per_cube_wrapper(b.expanded_depth_per_cube_wrapper),
            expanded_depth_per_cube(expanded_depth_per_cube_wrapper[0])
    {
        std::cout << "Copy constructor is called" << std::endl;
    }

    // Move constructor
    GridComputationWrapper(GridComputationWrapper&& other) noexcept : GridComputationWrapper() // initialize via default constructor, C++11 only
    {
        std::cout << "&& Move operator is called" << std::endl;
        // Vec::operator=(other);
        swapp(*this, other);
    }

    // Move assignment
    GridComputationWrapper & operator=(GridComputationWrapper &&other ) // Changed on my own from no & to && (from DevDat other to &&other)
    {
        std::cout << "Assignment operator is called" << std::endl;
        // Vec::operator=(other);
        swapp(*this, other);
        return *this;
    }

    // Copy Assignement
    GridComputationWrapper & operator=(const GridComputationWrapper& other )
    {
        std::cout << "Copy assignment operator is called" << std::endl;
        return *this = GridComputationWrapper(other);
    }

    // https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    friend void swapp(GridComputationWrapper& first, GridComputationWrapper& second) // nothrow
    {
        // enable ADL (not necessary in our case, but good practice)
        using std::swap;

        // by swapping the members of two objects,
        // the two objects are effectively swapped
        swap(first.expanded_cube_indices, second.expanded_cube_indices);
        swap(first.expanded_depth_per_cube_wrapper, second.expanded_depth_per_cube_wrapper);
        swap(first.expanded_depth_per_cube, second.expanded_depth_per_cube);
    } */

    void print_expanded_vectors()
    {
        auto i = 0;
        for (auto depth_index = 0; depth_index < expanded_cube_indices.dim_size(); depth_index++) {
            print_range(
                    "Expanded cube indices after filling with individual cube indices in depth " + std::to_string(i),
                    expanded_cube_indices[depth_index].begin(),
                    expanded_cube_indices[depth_index].end());
            i++;
        }
        print_range("Expanded depth per node", expanded_depth_per_cube.begin(),
                    expanded_depth_per_cube.end());
    }

    // For parent cube indices
    DevDatInt expanded_cube_indices;
    // For depths
    DevDatInt expanded_depth_per_cube_wrapper;
    DimensionIteratorInt &expanded_depth_per_cube;
};

#endif //PROGRAM_GRIDCOMPUTATIONWRAPPER_HPP
