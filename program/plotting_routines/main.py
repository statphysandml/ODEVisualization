import sys


def plot_hyperbolic_system():
    root_dir = "/data/"
    theory = "hyperbolic_system"
    config_dir = "visualization"

    relative_path = True

    from loading import FlowData
    loading = FlowData(
        theory=theory,
        config_dir=config_dir,
        root_dir=root_dir,
        relative_path=relative_path)
    # loading.reconstruct_vertices()

    data = loading.get_data()
    config_data = loading.get_config_data()

    path = loading.get_config_direction()

    separatrices = loading.get_separatrices("0")
    from loading import stream_plot_hyperbolic_system
    stream_plot_hyperbolic_system(data=data, config_data=config_data, separatrices=separatrices, name="1", path=path,
                                  title="Flow Field")


def plot_3D_hyperbolic_system():
    root_dir = "/data/"
    theory = "3D_hyperbolic_system"
    config_dir = "visualization_yz"

    relative_path = True

    from loading import FlowData
    loading = FlowData(
        theory=theory,
        config_dir=config_dir,
        root_dir=root_dir,
        relative_path=relative_path)
    # loading.reconstruct_vertices()

    data = loading.get_data()
    config_data = loading.get_config_data()

    path = loading.get_config_direction()

    from loading import stream_plot_3D_hyperbolic_system
    separatrices = loading.get_separatrices("0")
    stream_plot_3D_hyperbolic_system(data=data[:40000], config_data=config_data, separatrices=separatrices, name="1", path=path,
                                  title="Separatrices")
    separatrices = loading.get_separatrices("1")
    stream_plot_3D_hyperbolic_system(data=data[40000:80000], config_data=config_data, separatrices=separatrices,
                                     name="2", path=path,
                                     title="Separatrices")
    separatrices = loading.get_separatrices("2")
    stream_plot_3D_hyperbolic_system(data=data[80000:120000], config_data=config_data, separatrices=separatrices,
                                     name="3", path=path,
                                     title="Separatrices")


def plot_3D_hyperbolic_system_xy():
    root_dir = "/data/"
    theory = "3D_hyperbolic_system"
    config_dir = "visualization"

    relative_path = True

    from loading import FlowData
    loading = FlowData(
        theory=theory,
        config_dir=config_dir,
        root_dir=root_dir,
        relative_path=relative_path)
    # loading.reconstruct_vertices()

    data = loading.get_data()
    config_data = loading.get_config_data()

    path = loading.get_config_direction()

    from loading import stream_plot_3D_hyperbolic_system_xy
    separatrices = loading.get_separatrices("0")
    stream_plot_3D_hyperbolic_system_xy(data=data[:40000], config_data=config_data, separatrices=separatrices, name="1", path=path,
                                  title="Separatrices")
    separatrices = loading.get_separatrices("1")
    stream_plot_3D_hyperbolic_system_xy(data=data[40000:80000], config_data=config_data, separatrices=separatrices,
                                     name="2", path=path,
                                     title="Separatrices")
    separatrices = loading.get_separatrices("2")
    stream_plot_3D_hyperbolic_system_xy(data=data[80000:120000], config_data=config_data, separatrices=separatrices,
                                     name="3", path=path,
                                     title="Separatrices")


def plot_three_point_system():
    root_dir = "/data/"
    theory = "three_point_system"
    config_dir = "visualization"

    relative_path = True

    from loading import FlowData
    loading = FlowData(
        theory=theory,
        config_dir=config_dir,
        root_dir=root_dir,
        relative_path=relative_path)
    # loading.reconstruct_vertices()

    data = loading.get_data()
    config_data = loading.get_config_data()

    path = loading.get_config_direction()

    from loading import stream_plot
    # separatrices = loading.get_separatrices("0")
    # stream_plot(data=data[:6400], config_data=config_data, separatrices=separatrices, name="1", path=path,
    #             title="Fixed point for mh2=-0")

    separatrices = loading.get_separatrices("1")
    stream_plot(data=data[6400:12800], config_data=config_data, separatrices=separatrices, name="2", path=path,
                title="Fixed point for mh2= 0.0187292264855426")
    #
    # separatrices = loading.get_separatrices("2")
    # stream_plot(data=data[12800:], config_data=config_data, separatrices=separatrices, name="3", path=path,
    #           title="Fixed point at mh2 = -0.1645, Lam3 = -0.1641, gn = 0.5702")


def plot_four_point_system():
    root_dir = "/data/"
    theory = "four_point_system"
    config_dir = "visualization"

    relative_path = True

    from loading import FlowData
    loading = FlowData(
        theory=theory,
        config_dir=config_dir,
        root_dir=root_dir,
        relative_path=relative_path)
    # loading.reconstruct_vertices()

    data = loading.get_data()
    config_data = loading.get_config_data()

    path = loading.get_config_direction()

    from loading import stream_plot_four_point_system

    # separatrices = loading.get_separatrices("1")
    stream_plot_four_point_system(data=data, config_data=config_data, separatrices=None, name="2", path=path,
                title="")


if __name__ == '__main__':
    # plot_hyperbolic_system()
    # plot_three_point_system()
    plot_four_point_system()

    # import numpy as np
    #
    # x = np.random.randn(100)
    # y = np.random.randn(100)
    # z = np.sqrt(x * x + y * y)
    # #x = x/z
    # #y = y/z
    #
    # import matplotlib.pyplot as plt
    #
    # plt.scatter(x, y)
    # plt.show()
    # print(1)
