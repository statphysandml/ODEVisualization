//
// Created by lukas on 30.03.20.
//

#ifndef PROGRAM_TESTS_HPP
#define PROGRAM_TESTS_HPP

#include "../ode_solver/include/util/dev_dat.hpp"
#include "../ode_solver/include/extern/thrust_functors.hpp"

std::tuple< DevDatC > test_for_move_operators();

void print_system_info();

void devdat_basics_test();

#endif //PROGRAM_TESTS_HPP
