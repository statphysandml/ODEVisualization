#include "../include/config.h"

#include <odesolver/util/python_integration.hpp>

// #include "../flow_equations/three_point_system/three_point_system_jacobian.hpp"
#include "../include/fixed_point_search_t.hpp"


int main(int argc, char **argv) {
    // Initialize project dependent parameters
    param_helper::proj::set_relative_path_to_project_root_dir("../");

#ifdef PYTHON_BACKEND
    odesolver::util::initialize_python(PYTHON_SCRIPTS_PATH);
#endif

#ifndef GPU
    std::cout << "Running without GPU" << std::endl;
#else
    std::cout << "Running with GPU" << std::endl;
#endif

    // testing_devdat();
    // fixed_point_search();
    // fixed_points_search();
    find_fixed_points();
    // evaluate_fixed_points();

    // Finalization
#ifdef PYTHON_BACKEND
    odesolver::util::finalize_python();
#endif
    return 0;
}
