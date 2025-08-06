import numpy as np
from odesolver.coordinates import Coordinates

from lorentz_attractor import LorentzAttractor

from odesolver.recursive_search import RecursiveSearch

from odevisualizationlib.recursive_search import FixedPointCriterion


if __name__ == "__main__":
    pass

    # print(lorentz_attractor.json())

    # flow = lorentz_attractor(coordinates)
    #
    # flow = Coor([0.0 for _ in range(30)], 3)
    # lorentz_attractor(coordinates, flow)
    #
    # from odevisualization.modes import FixedPointSearch
    #

    #
    # fixed_point_search.evaluate("dynamic_memory")
    #
    # fixed_points = fixed_point_search.fixed_points_
    #
    # from odevisualization.utils import KMeansClustering
    # kmeans_clustering = KMeansClustering(20, 0.1, 1000)
    # fixed_points = kmeans_clustering.compute(fixed_points)
    #
    # fixed_point_search.write_to_file("./", "fixed_points")
    #
    # from odevisualization import CoordinateOperator
    #
    # coordinate_operator = CoordinateOperator(lorentz_attractor)
    #
    # velocities = coordinate_operator.evaluate()
    #
    # coordinate_operatr.evolve(...)
    #
    # from odevisualization import Visualization
    #
    # visualization = Visualization(
    #     [10, 1, 10],
    #     [[-12.0, 12.0], [-12.0, 12.0], [-1.0, 31.0]],
    #     [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
    #     lorentz_attractor
    # )
    #
    # velocities = visualization.eval()
    #
