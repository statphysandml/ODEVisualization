from lorentzattractor.lorentz_attractor import LorentzAttractor

# from odesolver.observers import TrajectoryObserver, IntersectionObserver


from odesolver.evolution import Evolution

from odesolver.coordinates import Coordinates

if __name__ == "__main__":
    lorentz_attractor = LorentzAttractor()

    coordinates = Coordinates.generate(dim=3, N=10, init_val=0.1)

    evolution = Evolution(flow_equations=lorentz_attractor)

    from odevisualizationlib.evolution.stepper import RungaKutta4

    stepper = RungaKutta4()

    print(coordinates)
    evolution.evolve(coordinates=coordinates, stepper=stepper, start_t=0.0, dt=0.01, n=10)

    print(coordinates)
    evolution.evolve(coordinates=coordinates, stepper=stepper, start_t=0.0, dt=-0.01, n=10)

    print(coordinates)

    evolution.evolve(coordinates=coordinates, stepper=stepper, start_t=0.0, end_t=0.1, dt=0.01)
    print(coordinates)

    # trajectories = lorentz_attractor.evolve(coordinates=coordinates, start_t = 0.0, end_t = 0.0, delta_t=0.01, step_size_type="constant", observer=trajectory_observer)
    #
    # coordinates = Coordinates.generate(dim=3, N=10, init_val=9.9)
    # trajectories = lorentz_attractor.evolve(coordinates=coordinates, start_t = 0.0, end_t = 0.0, delta_t=0.01, step_size_type="constant", observer=trajectory_observer)
    #
    #
    # intersection_observer = IntersectionObserver(boundaries=[[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], minimum_change_of_state=[1.0, 1.0, 1.0], minimum_delta=1e-6, maximum_flow_val=1e10, vicinity_distances=[0.1, 0.1, 0.1])
    #
    # coordinates = Coordinates.generate(dim=3, N=10, init_val=9.9)
    # intersection = lorentz_attractor.evolve(coordinates=coordinates, start_t = 0.0, end_t = 0.0, delta_t=0.01, step_size_type="constant", observer=intersection_observer)
    #
    # intersection_observer = IntersectionObserver(boundaries=[[-50.0, 50.0], [-50.0, 50.0]], minimum_change_of_state=[1.0, 1.0, 1.0], minimum_delta=1e-6, maximum_flow_val=1e10, vicinity_distances=[0.1, 0.1, 0.1])
    #
    # coordinates = Coordinates.generate(dim=3, N=10, init_val=9.9)
    # intersection, trajectories = lorentz_attractor.evolve(coordinates=coordinates, start_t = 0.0, end_t = 0.0, delta_t=0.01, step_size_type="constant", observer=[intersection_observer, trajectory_observer])


