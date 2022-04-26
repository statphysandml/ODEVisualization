//
// Created by kades on 8/5/19.
//

#ifndef PROJECT_EVOLUTION_OBSERVER_HPP
#define PROJECT_EVOLUTION_OBSERVER_HPP

#include "../extern/thrust_functors.hpp"
#include "param_helper/params.hpp"
#include "../flow_equation_interface/flow_equation_system.hpp"
#include "../odesolver/utilhelper_functions.hpp"
#include "../odesolver/util/header.hpp"
#include "../odesolver/util/dev_dat.hpp"
#include "../odesolver/util/monitor.hpp"

struct EvolutionObserver
{
    virtual void operator() (const odesolver::DevDatC &coordinates, cudaT t) = 0;
};

#endif //PROJECT_EVOLUTION_OBSERVER_HPP
