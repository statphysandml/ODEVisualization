import os

import numpy as np
import pandas as pd
import fnmatch

from figure_management.loading_figure_mode import loading_figure_mode
fma = loading_figure_mode("saving")


def json_to_file(file, data):
    import json
    with open(file + '.json', 'w') as outfile:
        json.dump(data, outfile, indent=4, separators=(',', ': '))


def json_from_file(file):
    import json
    with open(file + '.json') as data_file:
        return json.load(data_file)


class FlowData:
    def __init__(self, theory, config_dir, root_dir="/data/", relative_path=True):
        self.data = None
        self.separatrizes = None
        self.config_data = None

        self.indices_of_spanning_dimenions = None

        self.theory = theory
        self.config_dir = config_dir
        self.root_dir = root_dir
        self.relative_path = relative_path

        self.load_results()
        self.load_config_file()

    # Todo
    def reconstruct_vertices(self):
        num_per_dimension = np.array([len(fix_lambda) for fix_lambda in self.config_data["fix_lambdas"]])
        cum_num_per_dimension = np.cumprod(num_per_dimension)
        size_per_instance = np.prod(self.config_data["n_branches"])
        total_size = cum_num_per_dimension[-1] * size_per_instance
        total_number_of_sets = cum_num_per_dimension[-1]

        self.indices_of_spanning_dimenions = np.argwhere(np.array(self.config_data["n_branches"]) > 1).flatten()
        x_ranges = self.config_data['partial_lambda_ranges'][0]
        y_ranges = self.config_data['partial_lambda_ranges'][1]

        num_x = self.config_data["n_branches"][self.indices_of_spanning_dimenions[0]]
        x = np.tile(np.linspace(x_ranges[0], x_ranges[1] - (x_ranges[1] - x_ranges[0]) / num_x, num_x), int(total_size/num_x))

        num_y = self.config_data["n_branches"][self.indices_of_spanning_dimenions[1]]
        y = np.tile(np.repeat(np.linspace(y_ranges[0], y_ranges[1] - (y_ranges[1] - y_ranges[0]) / num_y, num_y), num_x), int(total_size/size_per_instance))

        vertices = dict()
        vertices[self.indices_of_spanning_dimenions[0]] = x
        vertices[self.indices_of_spanning_dimenions[1]] = y

        num = 0
        for idx, n_branch in enumerate(self.config_data["n_branches"]):
            if idx not in vertices.keys():
                repetition_factor = int(total_number_of_sets / cum_num_per_dimension[num]) * size_per_instance
                vertices[idx] = np.tile(np.repeat(self.config_data["fix_lambdas"][num], repetition_factor), int(cum_num_per_dimension[num]/num_per_dimension[num])) # Todo: -> equal to cum_num_per_dimension[num-1]
                print(len(vertices[idx]))
                num += 1

        vertices = pd.DataFrame(vertices)
        # vertices = {num: [] for num in range(self.config_data["dim"])}
        # vertices[0] = np.tile(self.config_data["fix_lambdas"][0], 1).repeat(repetitions[-1]/ len(self.config_data["fix_lambdas"][0])).repeat(size_per_instance)
        # vertices[1] = np.tile(self.config_data["fix_lambdas"][1], 1).repeat(repetitions[-1]/ len(self.config_data["fix_lambdas"][1])).repeat(size_per_instance)
        # vertices[2] = np.tile(self.config_data["fix_lambdas"][2], int(repetitions[-2]/repetitions[-3])).repeat(size_per_instance)
        pass

    def load_results(self):
        filenames = ["vertices", "velocities"]
        self.data = [self.load_results_from_path(
            filename=filename,
            theory=self.theory,
            config_dir=self.config_dir,
            root_dir=self.root_dir,
            relative_path=self.relative_path
        ) for filename in filenames]
        self.data = pd.concat(self.data, axis=1, keys=filenames)

        # separatrices_files = self.get_files_from_extension(
        #     base_string="separatrices",
        #     theory=self.theory,
        #     config_dir=self.config_dir,
        #     root_dir=self.root_dir,
        #     relative_path=self.relative_path
        # )
        #
        # self.separatrizes = [self.load_results_from_path(
        #     filename=file,
        #     theory=self.theory,
        #     config_dir=self.config_dir,
        #     root_dir=self.root_dir,
        #     relative_path=self.relative_path
        # ) for file in separatrices_files]
        # self.separatrizes = pd.concat(self.separatrizes, axis=0, keys=separatrices_files)

    def get_data(self):
        return self.data

    def get_config_data(self):
        return self.config_data

    def get_separatrices(self, appendix):
        return self.separatrizes.loc["separatrices_" + appendix]

    def load_config_file(self):
        path = self.get_config_direction()
        global_parameters = json_from_file(path + "/global_parameters")
        config_data = json_from_file(self.root_dir + "/data/" + self.theory + "/" + self.config_dir + "/config")
        self.config_data = {**global_parameters, ** config_data}

    def get_config_direction(self):
        # cwd = os.getcwd()
        return self.root_dir + "/flow_equations/" + self.theory + "/"

    @staticmethod
    def load_results_from_path(filename, theory, config_dir, root_dir="/data/", relative_path=True):
        cwd = os.getcwd()
        data = pd.read_csv(root_dir + "/data/" + theory + "/" + config_dir + "/" + filename + ".dat", delimiter=" ", header=None)
        del data[len(data.columns) - 1]
        data.index.name = filename
        return data

    @staticmethod
    def get_files_from_extension(base_string, theory, config_dir, root_dir="/data/", relative_path=True):
        cwd = os.getcwd()
        filenames = []
        directory = cwd + "/" + root_dir + "/" + theory + "/" + config_dir + "/"
        for file in os.listdir(directory):
            if fnmatch.fnmatch(file, base_string + "_*.dat") and os.stat(directory + file).st_size != 0:
                filenames.append(file[:-4])
        filenames.sort()
        return filenames


def stream_plot_hyperbolic_system(data, config_data, separatrices, name, path, title):
    col_x = 0
    col_y = 1

    dim_x = config_data["n_branches"][col_x]
    dim_y = config_data["n_branches"][col_y]

    X = data["vertices"][col_x].values.reshape(dim_y, dim_x)
    Y = data["vertices"][col_y].values.reshape(dim_y, dim_x)

    U = data["velocities"][col_x].values.reshape(dim_y, dim_x)
    V = data["velocities"][col_y].values.reshape(dim_y, dim_x)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig, axes = fma.newfig(1.0, ratio=1)
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])
    axes.streamplot(X, Y, U, V, density=[3, 3], linewidth=0.3, arrowsize=0.3)

    if len(separatrices) > 0:
        axes.scatter(separatrices[0], separatrices[1], s=2.0, c="black")
    # axes.set_title('Flowfield for the Four-Point System for Lam4=-0.1082, g3=0.63998 and g4=0.553988')
    # axes.set_title("Hyperbolic System")
    axes.set_xlabel('x')
    axes.set_ylabel('y')

    y, z = config_data["fixed_points"]["fixed_points"][0]
    axes.scatter(y, z, s=12.0, c="red", label="Fixed point")

    axes.set_xlim(-2, 6)
    axes.set_ylim(-5, 3)

    plt.tight_layout()

    # axes.legend()

    fma.savefig(path, "flow_field" + name + "_pure")
    plt.close()


def stream_plot_3D_hyperbolic_system(data, config_data, separatrices, name, path, title):
    col_x = 1
    col_y = 2

    dim_x = config_data["n_branches"][col_x]
    dim_y = config_data["n_branches"][col_y]

    X = data["vertices"][col_x].values.reshape(dim_y, dim_x)
    Y = data["vertices"][col_y].values.reshape(dim_y, dim_x)

    U = data["velocities"][col_x].values.reshape(dim_y, dim_x)
    V = data["velocities"][col_y].values.reshape(dim_y, dim_x)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig, axes = fma.newfig(1.0, ratio=1)
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])

    every = 10
    axes.streamplot(X, Y, U, V, density=[3, 3], linewidth=0.3, arrowsize=0.3)

    if len(separatrices) > 0:
        axes.scatter(separatrices[0][::every], separatrices[1][::every], s=1.0)

    # axes.set_title('Flowfield for the Four-Point System for Lam4=-0.1082, g3=0.63998 and g4=0.553988')
    axes.set_title(title)
    axes.set_xlabel('y')
    axes.set_ylabel('z')

    # axes.scatter()

    x, y, z = config_data["fixed_points"]["fixed_points"][0]
    axes.scatter(y, z, s=5.0, label="Fixed point")

    plt.tight_layout()

    axes.legend()

    fma.savefig(path, "flow_field" + name)
    plt.close()


def stream_plot_3D_hyperbolic_system_xy(data, config_data, separatrices, name, path, title):
    col_x = 0
    col_y = 1

    dim_x = config_data["n_branches"][col_x]
    dim_y = config_data["n_branches"][col_y]

    X = data["vertices"][col_x].values.reshape(dim_y, dim_x)
    Y = data["vertices"][col_y].values.reshape(dim_y, dim_x)

    U = data["velocities"][col_x].values.reshape(dim_y, dim_x)
    V = data["velocities"][col_y].values.reshape(dim_y, dim_x)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig, axes = fma.newfig(1.0, ratio=1)
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])

    every = 1
    axes.streamplot(X, Y, U, V, density=[3, 3], linewidth=0.3, arrowsize=0.3)

    if len(separatrices) > 0:
        axes.scatter(separatrices[0][::every], separatrices[1][::every], s=1.0)

    # axes.set_title('Flowfield for the Four-Point System for Lam4=-0.1082, g3=0.63998 and g4=0.553988')
    axes.set_title(title)
    axes.set_xlabel('x')
    axes.set_ylabel('y')

    # axes.scatter()

    x, y, z = config_data["fixed_points"]["fixed_points"][0]
    axes.scatter(x, y, s=5.0, label="Fixed point")

    plt.tight_layout()

    axes.legend()

    fma.savefig(path, "flow_field" + name)
    plt.close()


def stream_plot(data, config_data, separatrices, name, path, title):
    col_x = 1
    col_y = 2

    dim_x = config_data["n_branches"][col_x]
    dim_y = config_data["n_branches"][col_y]

    X = data["vertices"][col_x].values.reshape(dim_y, dim_x)
    Y = data["vertices"][col_y].values.reshape(dim_y, dim_x)

    U = data["velocities"][col_x].values.reshape(dim_y, dim_x)
    V = data["velocities"][col_y].values.reshape(dim_y, dim_x)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # w = 3
    # v = 2
    # Y, X = np.mgrid[-w:w:100j, -v:v:80j]
    # U = -1 - X ** 2 + Y
    # V = 1 + X - Y ** 2
    # speed = np.sqrt(U ** 2 + V ** 2)

    fig, axes = fma.newfig(1.0, ratio=1)
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])
    every = 10
    axes.streamplot(X, Y, U, V, density=[3, 3], linewidth=0.3, arrowsize=0.3)

    # if len(separatrices) > 0:
    #     axes.scatter(separatrices[0][::every], separatrices[1][::every], s=1.0)
    # axes.set_title('Flowfield for the Four-Point System for Lam4=-0.1082, g3=0.63998 and g4=0.553988')
    # axes.set_title(title)
    axes.set_xlabel('Lam3')
    axes.set_ylabel('gn')

    # axes.scatter()

    # x, y, z = config_data["fixed_points"]["fixed_points"][0]
    # axes.scatter(y, z, s=12.0, label="mh2 = 0.000")
    # 
    # x, y, z = config_data["fixed_points"]["fixed_points"][1]
    # axes.scatter(y, z, s=12.0, label="mh2 = 0.187")
    # 
    # x, y, z = config_data["fixed_points"]["fixed_points"][2]
    # axes.scatter(y, z, s=12.0, label="mh2 = -0.1645")

    axes.set_xlim(-1.8, 0.9)
    axes.set_ylim(-0.61, 1.0)
    plt.tight_layout()

    # axes.legend()

    fma.savefig(path, "flow_field_pure" + name)
    plt.close()


def get_exp_norm_and_levs(lev_min, lev_max, lev_num):
    lev_exp = np.linspace(np.log10(lev_min), np.log10(lev_max), lev_num)  # 0.5
    levs = np.power(10, lev_exp)

    from matplotlib import colors
    return colors.LogNorm(), levs


def stream_plot_four_point_system(data, config_data, separatrices, name, path, title):
    col_x = 0
    col_y = 1

    dim_x = config_data["n_branches"][col_x]
    dim_y = config_data["n_branches"][col_y]

    X = data["vertices"][col_x].values.reshape(dim_y, dim_x)
    Y = data["vertices"][col_y].values.reshape(dim_y, dim_x)

    U = data["velocities"][col_x].values.reshape(dim_y, dim_x)
    V = data["velocities"][col_y].values.reshape(dim_y, dim_x)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # w = 3
    # v = 2
    # Y, X = np.mgrid[-w:w:100j, -v:v:80j]
    # U = -1 - X ** 2 + Y
    # V = 1 + X - Y ** 2
    # speed = np.sqrt(U ** 2 + V ** 2)

    fig, axes = fma.newfig(0.8, ratio=0.8)
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])
    every = 10
    axes.streamplot(X, Y, U, V, density=[2.4, 2.4], linewidth=0.5, arrowsize=0.3)

    # if len(separatrices) > 0:
    #     axes.scatter(separatrices[0][::every], separatrices[1][::every], s=1.0)
    # axes.set_title('Flowfield for the Four-Point System for Lam4=-0.1082, g3=0.63998 and g4=0.553988')
    # axes.set_title(title)
    axes.set_xlabel('$\mu$')
    axes.set_ylabel('$\lambda_3$')

    # axes.scatter()

    # x, y, z = config_data["fixed_points"]["fixed_points"][0]
    # axes.scatter(y, z, s=12.0, label="mh2 = 0.000")
    #
    # x, y, z = config_data["fixed_points"]["fixed_points"][1]
    # axes.scatter(y, z, s=12.0, label="mh2 = 0.187")
    #
    # x, y, z = config_data["fixed_points"]["fixed_points"][2]
    # axes.scatter(y, z, s=12.0, label="mh2 = -0.1645")

    # axes.scatter([-0.226227508697766], [-0.0603059942535251], s=12.0, c="red")

    axes.set_xlim(-3.0, 1.0)
    axes.set_ylim(-1.25, 0.75)
    plt.tight_layout()

    # axes.legend()

    fma.savefig(path, "flow_field_small")
    plt.close()


def stream_plot_four_point_system_as(data, config_data, separatrices, name, path, title):
    col_x = 0
    col_y = 1

    dim_x = config_data["n_branches"][col_x]
    dim_y = config_data["n_branches"][col_y]

    X = data["vertices"][col_x].values.reshape(dim_y, dim_x)
    Y = data["vertices"][col_y].values.reshape(dim_y, dim_x)

    U = data["velocities"][col_x].values.reshape(dim_y, dim_x)
    V = data["velocities"][col_y].values.reshape(dim_y, dim_x)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # w = 3
    # v = 2
    # Y, X = np.mgrid[-w:w:100j, -v:v:80j]
    # U = -1 - X ** 2 + Y
    # V = 1 + X - Y ** 2
    # speed = np.sqrt(U ** 2 + V ** 2)

    fig, axes = fma.newfig(1.0, ratio=1)
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])
    every = 10
    # axes.streamplot(X, Y, U, V, density=[3, 3], linewidth=0.3, arrowsize=0.3)

    Z = np.sqrt(np.power(U, 2) + np.power(V, 2))
    Z[np.isnan(Z)] = 1000 # np.max(Z[~np.isnan(Z)])
    Z[Z > 1000] = 1000

    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator

    levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    norm, levels = get_exp_norm_and_levs(Z.min(), Z.max(), 100)

    scaling = 5
    cf = axes.contourf(X, Y, Z, levels=levels, norm=norm, cmap=cmap)
    fig.colorbar(cf, ax=axes)

    # if len(separatrices) > 0:
    #     axes.scatter(separatrices[0][::every], separatrices[1][::every], s=1.0)
    # axes.set_title('Flowfield for the Four-Point System for Lam4=-0.1082, g3=0.63998 and g4=0.553988')
    # axes.set_title(title)
    axes.set_xlabel('Mu')
    axes.set_ylabel('Lam3')

    # axes.scatter()

    # x, y, z = config_data["fixed_points"]["fixed_points"][0]
    # axes.scatter(y, z, s=12.0, label="mh2 = 0.000")
    #
    # x, y, z = config_data["fixed_points"]["fixed_points"][1]
    # axes.scatter(y, z, s=12.0, label="mh2 = 0.187")
    #
    # x, y, z = config_data["fixed_points"]["fixed_points"][2]
    # axes.scatter(y, z, s=12.0, label="mh2 = -0.1645")

    # axes.scatter([-0.226227508697766], [-0.0603059942535251], s=12.0, c="red")

    axes.set_xlim(-3.0, 2.0)
    axes.set_ylim(-2.0, 1.0)
    plt.tight_layout()

    # axes.legend()

    fma.savefig(path, "abs_length_field_pure" + name)
    plt.close()
