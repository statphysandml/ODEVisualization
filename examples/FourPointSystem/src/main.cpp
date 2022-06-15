#include "../include/fixed_point_search_t.hpp"

int main(int argc, char **argv) {
    // Initialize project dependent parameters
    param_helper::proj::set_relative_path_to_project_root_dir("../");

#ifndef GPU
    std::cout << "Running without GPU" << std::endl;
#else
    std::cout << "Running with GPU" << std::endl;
#endif

    // fixed_point_search();
    // fixed_points_search();
    find_fixed_points();

    return 0;
}
