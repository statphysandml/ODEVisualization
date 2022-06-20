import numpy as np

from odesolver.coordinates import Coordinates
from odesolver.jacobians import Jacobians

if __name__ == "__main__":
    coordinates = Coordinates.generate(dim=9, N=10, init_val=9.9)

    jacobians = Jacobians.from_vfcoor(elements=coordinates)

    jacobians = Jacobians.from_flattened_data(elements=[1.0] * 27, dim=3)
    
    jacobians = Jacobians.from_data(elements=[[1.0] * 9] * 4)
    # print(jacobians.elements())

    jacobians.eval()

    print(jacobians.jacobian(idx=1))
    print(jacobians.eigenvectors(idx=1))
    print(jacobians.eigenvalues(idx=1))