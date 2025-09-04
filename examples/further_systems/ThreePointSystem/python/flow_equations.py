from odesolver.coordinates import Coordinates
from three_point_system import ThreePointSystem


if __name__ == "__main__":
    coordinates = Coordinates.generate(dim=3, N=10, init_val=0.0)

    coordinates[:, 0] = [0.1, 0.2, 0.2]
    coordinates[:, 1] = [0.1, 0.3, 0.1]
    coordinates[:, 2] = [0.3, 0.2, 0.3]

    three_point_system = ThreePointSystem()
    print(three_point_system.dim)
    print(three_point_system.model)
    print(three_point_system.flow_variable)
    print(three_point_system.flow_parameters)

    flow = three_point_system.flow(coordinates=coordinates)
    jacobians = three_point_system.jacobians(coordinates=coordinates)

    print(flow)