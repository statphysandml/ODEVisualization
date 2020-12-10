//
// Created by lukas on 26.02.19.
//

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cassert>

#include "../include/executer.hpp"
#include "../include/extern/thrust_functors.hpp"
#include "../include/util/dev_dat.hpp"

#include "../../examples/tests.hpp"

#include "../../examples/examples_without_param_files.hpp"

#include "../../examples/visualisation.hpp"
#include "../../examples/fixed_point_search.hpp"
#include "../../examples/evaluate.hpp"
// #include "../../examples/jacobian.hpp"
#include "../../examples/evolve.hpp"
#include "../../examples/evolve_on_condition.hpp"

#include <Python.h>

void custom_main();

// ToDo: Write dummy config file generators
// ToDo: Think about useful usage of mode in different Main Modules - Add error message if appended parameters do not coincide with considered mode
// ToDo: Introduce developer_mode variable and switch between include files for custom systems...and all systems...before compilation

/* ToDo Next:
 * - Comment code with examples and check for modularity (clearify if existing)
 * - Recognize divergencies and abort further computations
 * - Check for correct deletion of objects
 * - Introduce smart pointers
 * - How much computation is overhead??
 * https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
*/


/* Main modules describe major operations based on a given set of ordinary differential equations:
 * 1) Fixed Point Search - Search and find fixed points
 * 2) Visualization- Different visualization functions
 * 3) Coordinate Operator - Perform different operations for given operators */

/* Helper modules - are used by main modules to implement the different functionalities
 * 1) Hypercubes - Class that represents hypercubes
 * 2) Evolution - Class that enables evolution of a given DevDatC vector
 * 3) EvolutionObserver - Class that enables observations during a transformation of a DevDatC vector */

/* Member helper modules - are used by the respective main module to implement a specific functionality -
 * these are only used by this particular main module class, in contrast to main modules - Member helper modules consist
 * of a parameter class and respective functions in their main module class */

/* Helper modules which are designed for specific main function of main modules have their own parameter class for a
 * possible storage of these parameters */


// Might be useful https://github.com/kigster/cmake-project-template


#include "param_helper/params.hpp"

int main(int argc, char **argv)
{
    Py_Initialize();
    PyRun_SimpleString("import sys\n" "import os");
    PyRun_SimpleString("sys.path.append( os.path.dirname(os.getcwd()) + '/../../program/')");
    PyRun_SimpleString("sys.path.append( os.path.dirname(os.getcwd()) + '/../../program/plotting_routines/')");
    PyRun_SimpleString("sys.path.append( os.path.dirname(os.getcwd()) + '/../../program/plotting_routines/plotting_environment/')");
    PyRun_SimpleString("print('Running python in ' + os.path.dirname(os.getcwd()) + '/../../program/plotting_routines/')");

    if(argc > 1)
        run_from_file(argc, argv);
    else
        custom_main();

    Py_Finalize();
}

void custom_main()
{
    // fixed_points_search();
    // fixed_point_search();

    // evaluate_velocities_and_jacobians_of_coordinates();
    // evaluate_velocities_and_jacobians();

    // evolve();

    // evolve_conditional_range();
    // evolve_conditional_intersection()

    // generate_fixed_point_search_parameters();
    // add_clustering_of_solution_to_fixed_point_search();
    // search_fixed_points();

    generate_visualization_parameters();
}
