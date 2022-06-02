#include "../../include/observers/tracking_observer.hpp"

void TrackingObserver::operator() (const odesolver::DevDatC &coordinates, cudaT t)
{
    os << t << " ";
    print_range_in_os(coordinates.begin(), coordinates.end(), os);
    os << std::endl;
}