//
// Created by lukas on 15.09.19.
//

#ifndef PROGRAM_TRACKING_OBSERVER_HPP
#define PROGRAM_TRACKING_OBSERVER_HPP

#include "evolution_observer.hpp"

struct TrackingObserver : public EvolutionObserver
{
public:
    TrackingObserver(std::ofstream &os_) : os(os_)
    {}

    void operator() (const odesolver::DevDatC &coordinates, cudaT t) override;

    static std::string name() {  return "tracking_observer";  }

    std::ofstream &os;
};

#endif //PROGRAM_TRACKING_OBSERVER_HPP
