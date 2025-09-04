import numpy as np
from odesolver.coordinates import Coordinates

from three_point_system import ThreePointSystem

from odesolver.recursive_search import RecursiveSearch
from odesolver.kmeans_clustering import KMeansClustering

from odesolver.fixed_point_criterion import FixedPointCriterion


if __name__ == '__main__':
    three_point_system = ThreePointSystem()

    fixed_point_criterion = FixedPointCriterion()

    recursive_fixed_point_search = RecursiveSearch(
        maximum_recursion_depth=18,
        n_branches_per_depth=[[10, 10, 10]] + [[2, 2, 2]] * 17,
        variable_ranges=[[-0.9, 0.9], [-1.8, 0.9], [-0.61, 1.0]],
        criterion=fixed_point_criterion,
        flow_equations=three_point_system,
        number_of_cubes_per_gpu_call=100,
        maximum_number_of_gpu_calls=100000
    )
    recursive_fixed_point_search.eval("dynamic")

    fixed_points = recursive_fixed_point_search.solutions()

    fixed_point_cube_index_path = recursive_fixed_point_search.solutions("cube_indices")

    # grid_computation.project_cube_indices_on_center_vertices()

    print(fixed_points.transpose())

    kmeans_clustering = KMeansClustering(
        maximum_expected_number_of_clusters=10,
        upper_bound_for_min_distance=0.01,
        maximum_number_of_iterations=1000
    )

    kmeans_clustering.eval(fixed_points)
