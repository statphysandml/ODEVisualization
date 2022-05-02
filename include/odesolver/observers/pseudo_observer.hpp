//
// Created by lukas on 02.04.20.
//

#ifndef PROGRAM_PSEUDO_OBSERVER_HPP
#define PROGRAM_PSEUDO_OBSERVER_HPP

#include "evolution_observer.hpp"

struct PseudoObserver : public EvolutionObserver
{
public:
    PseudoObserver()
    {}

    void operator() (const odesolver::DevDatC &coordinates, cudaT t) override
    {}

    static std::string name() {  return "psuedo_observer";  }
};

#endif //PROGRAM_PSEUDO_OBSERVER_HPP
