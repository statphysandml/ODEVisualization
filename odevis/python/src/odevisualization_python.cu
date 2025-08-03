#include "../include/odevisualization_python.hpp"


PYBIND11_MODULE(odevisualizationlib, m)
{
    // odesolver::pybind::init_functions(m);
    odesolver::pybind::init_devdat(m);
    odesolver::pybind::init_evolution(m);
    odesolver::pybind::init_flow_equations(m);
    odesolver::pybind::init_modes(m);
    odesolver::pybind::init_recursive_search(m);

    m.doc() = "Python Bindings for MCMCSimulationLib";
}
