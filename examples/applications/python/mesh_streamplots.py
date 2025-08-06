import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from odesolver.coordinates import Coordinates

from lorentz_attractor import LorentzAttractor

from odesolver.mesh import Mesh

if __name__ == '__main__':
    lorentz_attractor = LorentzAttractor()

    fixed_points = np.array([[-9.15527344e-06, -9.15527344e-06, -1.22070313e-05],
                             [-8.48529968e+00, -8.48528137e+00,  2.70000122e+01],
                             [ 8.48528137e+00,  8.48528137e+00,  2.70000122e+01]])

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(nrows=3, ncols=3, height_ratios=[1, 1, 1])

    mesh = Mesh(n_branches=[3, 5], variable_ranges=[[-24.0, 24.0], [-32.0, 32.0]])
    Y, X = mesh.mgrid()
    YY, XX = np.mgrid[-32.0:32.0:5j, -24.0:24.0:3j]
    assert np.all(Y == YY) and np.all(X == XX)

    mesh = Mesh(n_branches=[3, 5, 9], variable_ranges=[[-24.0, 24.0], [-32.0, 32.0], [-17.0, 45.0]])
    vertices = mesh.vertices()
    Z, Y, X = mesh.mgrid(data=vertices)
    ZZ, YY, XX = np.mgrid[-17.0:45.0:9j, -32.0:32.0:5j, -24.0:24.0:3j]
    assert np.all(Y == YY) and np.all(X == XX) and np.all(Z == ZZ)

    mesh = Mesh(n_branches=[13, 17, 1], variable_ranges=[[-24.0, 24.0], [-32.0, 32.0]], fixed_variables=fixed_points[:, 2].reshape(-1, 1))
    Y, X = mesh.mgrid(fixed_variable_idx=0)
    for i in range(3):
        vertices = mesh.vertices(fixed_variable_idx=i)
        flow = lorentz_attractor.flow(vertices)
        V, U = mesh.mgrid(data=flow)
        ax0 = fig.add_subplot(gs[0, i])
        ax0.streamplot(X, Y, U, V, density=[1.0, 1.0], linewidth=1.0, arrowsize=0.7)
        ax0.scatter(fixed_points[i, 0], fixed_points[i, 1], s=20, c="black", marker="x")
        ax0.set_title('Varying Density ' + str(i) + " in dimensions 1 and 2")

    mesh = Mesh(n_branches=[13, 1, 17], variable_ranges=[[-24.0, 24.0], [-17.0, 47.0]], fixed_variables=fixed_points[:, 1].reshape(-1, 1))
    Y, X = mesh.mgrid(fixed_variable_idx=0)
    for i in range(3):
        vertices = mesh.vertices(fixed_variable_idx=i)
        flow = lorentz_attractor.flow(vertices)
        V, U = mesh.mgrid(data=flow)
        ax0 = fig.add_subplot(gs[1, i])
        ax0.streamplot(X, Y, U, V, density=[1.0, 1.0], linewidth=1.0, arrowsize=0.7)
        ax0.scatter(fixed_points[i, 0], fixed_points[i, 2], s=20, c="black", marker="x")
        ax0.set_title('Varying Density ' + str(i) + " in dimensions 1 and 3")

    mesh = Mesh(n_branches=[1, 17, 17], variable_ranges=[[-32.0, 32.0], [-17.0, 47.0]], fixed_variables=fixed_points[:, 0].reshape(-1, 1))
    Y, X = mesh.mgrid(fixed_variable_idx=0)
    for i in range(3):
        vertices = mesh.vertices(fixed_variable_idx=i)
        flow = lorentz_attractor.flow(vertices)
        V, U = mesh.mgrid(data=flow)
        ax0 = fig.add_subplot(gs[2, i])
        ax0.streamplot(X, Y, U, V, density=[1.0, 1.0], linewidth=1.0, arrowsize=0.7)
        ax0.scatter(fixed_points[i, 1], fixed_points[i, 2], s=20, c="black", marker="x")
        ax0.set_title('Varying Density ' + str(i) + " in dimensions 2 and 3")

    plt.tight_layout()
    plt.show()

    # fig, ax = plt.subplots(figsize=(10, 7))
    # # X: "Row by row" -> dim: (n_rows, n_cols) = (n_y, n_x)
    # # Y: "Row by row" -> dim: (n_rows, n_cols) = (n_y, n_x)
    # ax.streamplot()


    # np.array(vertices.data).reshape(3, 9, 13, 13)

    # recursive_search = RecursiveSearch(
    #   maximum_recursion_depth,
    #   n_branches_per_depth,
    #   variable_ranges,
    #   flow_equations,
    #   criterion=...) (-> determine_potential_fixed_points,)

    # recursive_search.eval("dynamic")

    # recursive_search.solutions(cube_indices vs coordinates)

    # grid_computation = GridComputation(n_branches, lambda_ranges)
    # grid_computation.compute_center_vertices()
    # grid_computation.compute_reference_vertices()
    # grid_computation.compute_cube_vertices()

    # grid_computation.project_cube_indices_on_center_vertices()
    # ...

    # dynamic_grid_computation ...


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
