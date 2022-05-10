//
// Created by kades on 8/5/19.
//

#ifndef PROJECT_EVOLUTION_OBSERVER_HPP
#define PROJECT_EVOLUTION_OBSERVER_HPP

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/helper_functions.hpp>
#include <odesolver/util/thrust_functors.hpp>
#include <odesolver/flow_equations/flow_equation_system.hpp>

struct EvolutionObserver
{
    virtual void operator() (const odesolver::DevDatC &coordinates, cudaT t) = 0;
};

#endif //PROJECT_EVOLUTION_OBSERVER_HPP
