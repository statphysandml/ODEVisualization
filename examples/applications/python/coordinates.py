import numpy as np
from odesolver.coordinates import Coordinates


if __name__ == "__main__":
    coordinates = Coordinates.generate(dim=3, N=10, init_val=9.9)
    coordinates[0] = 4.0  # Set dim 0
    coordinates[0] = [1.0, 3.5, 2.0, 2.0, 4.0, 1.0, 3.0, 2.0, 2.0, 4.0]   # Set dim 0
    coordinates[0, :] = 2.0  # Equivalent to coordinates[0] ...
    coordinates[0, :] = [1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 0.0]  # Equivalent to coordinates[0] ...
    coordinates[:, 2] = 3.0  # Set element 2
    coordinates[:, 1] = [1.0, 3.0, 2.0]  # Set element 1

    coordinates[0, 1:-1] = -1.3  # Set sub dim 0
    coordinates[0, 1:-1] = [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0]  # Set sub dim 0
    coordinates[0:-1, 1] = -8  # Set sub element 1
    coordinates[0:-1, 1] = [1.0, 2.0]  # Set sub element 1

    coordinates[:, 1:-1] = -3.9
    coordinates[:, 1:-1] = [[-10.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0], [-11.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0], [-12.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0]]

    coordinates[1:3, :] = 2.0
    coordinates[1:3, :] = [[-9.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 9.0, 10.0, 1.0], [-11.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, -8.0, -4.0, -5.0]]

    coordinates[0:2, -4:] = [[9.0, 2.0, 2.0, -9.0], [-3.0, 2.0, 2.0, 3.0]]

    # Special cases
    coordinates[:, 1:-1] = [-10.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0]
    coordinates[1:3, :] = [1.0, -1.0]

    coordinates[:, :] = 0.0
    # coordinates[:, :] = [-10.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0, -1.0, -1.0]

    coordinates[:, :] = [[-10.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0, -1.0, -1.0], [-11.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0, -1.0, -1.0], [-12.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0, -1.0, -1.0]]

    coordinates[1, 1] = 4.0

    print(coordinates)
    print(coordinates[1])
    print(coordinates[0, :])
    print(coordinates[:, 1])
    print(coordinates[0, 4:])
    print(coordinates[-2:, 1])

    print(coordinates[:, 1:-1])
    print(coordinates[1:3, :])
    print(coordinates[0:2, -4:])
    print(coordinates[:, :])
    print(coordinates[1, 1])


    coordinates[1, 1:3] = [2.0, 3.0]

    coordinates.fill_data_in_dim(1, np.arange(0, 10))
    coordinates[1, :] = np.arange(0, 10)

    coordinates[1, 1:8] = np.arange(0, 7)

    coordinates = Coordinates.from_data([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    print(coordinates.transpose())

    coordinates = Coordinates.from_flattened_data(data=np.arange(0, 150), dim=5)
    print(coordinates)

    print("Dim", coordinates.dim())
    print("Size", coordinates.size())
    print("Shape", coordinates.shape)

    print("Coordinates in dim=0", coordinates.data_in_dim(dim=0))

    transposed_coordinates = coordinates.transpose()
    print(transposed_coordinates)

    coordinates.transpose(in_place=True)
    print(coordinates)

