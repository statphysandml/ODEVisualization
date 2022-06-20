#include "../include/dev_dat_t.hpp"
#include "../include/evolution_t.hpp"
#include "../include/flow_equations_t.hpp"
#include "../include/flow_observer_t.hpp"
#include "../include/fixed_point_search_t.hpp"
#include "../include/jacobians_t.hpp"
#include "../include/partial_ranges_t.hpp"
#include "../include/separatrizes_t.hpp"


int main(int argc, char **argv) {
    // Initialize project dependent parameters
    param_helper::proj::set_relative_path_to_project_root_dir("../");

#ifndef GPU
    std::cout << "Running without GPU" << std::endl;
#else
    std::cout << "Running with GPU" << std::endl;
#endif

    testing_devdat();
    find_fixed_points();
    compute_flow_t();

    find_fixed_points();
    partial_ranges_t();
    partial_ranges_t2();
    
    jacobians_t();
    evolution_t();

    no_change_t();
    divergent_flow_t();
    out_of_range_t();
    out_of_range_t2();
    intersection_observer_t();
    trajectory_observer_t();
    
    evolution_observer_t();
    conditional_range_observer_t();

    separatrizes_t();

    return 0;
}
