from odesolver.coordinates import Coordinates
from lorentz_attractor import LorentzAttractor


if __name__ == "__main__":
    coordinates = Coordinates.generate(dim=3, N=10, init_val=9.9)

    coordinates[:, 0] = [0, 0, 0]
    coordinates[:, 1] = [2, 1, 0]

    lorentz_attractor = LorentzAttractor()
    print(lorentz_attractor.dim)
    print(lorentz_attractor.model)
    print(lorentz_attractor.flow_variable)
    print(lorentz_attractor.flow_parameters)

    flow = lorentz_attractor.flow(coordinates=coordinates)
    jacobians = lorentz_attractor.jacobians(coordinates=coordinates)

    print(flow)