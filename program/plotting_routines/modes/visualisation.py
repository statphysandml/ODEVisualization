import sys
import numpy as np

from figure_management.loading_figure_mode import loading_figure_mode
fma = loading_figure_mode("saving")


def visualization(theory, project_directory, mode_directory, relative_path):
    from loading import FlowData
    loading = FlowData(
        theory=theory,
        config_dir=mode_directory,
        root_dir=project_directory,
        relative_path=relative_path
    )

    data = loading.get_data()
    config_data = loading.get_config_data()

    path = loading.get_config_direction()

    # ToDo Implement possiblity that fix_lambdas is not defined and that fix_lambdas is a list of lists
    fixed_vertex_indices = np.argwhere(np.array(config_data["n_branches"]) == 1)
    axes_to_plotted = np.argwhere(np.array(config_data["n_branches"]) != 1).flatten()
    title = ' '.join([item.capitalize() for item in theory.split(sep="_")])

    if config_data["fix_lambdas"] is None:
        stream_plot(data=data, config_data=config_data, axes_to_be_plotted=axes_to_plotted,
                    name=theory,
                    path=loading.root_dir + "/data/" + theory + "/" + loading.config_dir + "/", title=title)
    else:
        for fixed_val in config_data["fix_lambdas"][0]:
            mask = data.vertices[fixed_vertex_indices[0]] == fixed_val
            stream_plot(data=data.iloc[mask.values.flatten()], config_data=config_data, axes_to_be_plotted=axes_to_plotted, name=theory + "_{:.3f}".format(fixed_val),
                        path=loading.root_dir + "/data/" + theory + "/" + loading.config_dir + "/", title=title)


def stream_plot(data, config_data, axes_to_be_plotted, name, path, title, separatrices=None):
    dim_x, dim_y = np.array(config_data["n_branches"])[axes_to_be_plotted]
    axes_names = np.array(config_data["explicit_functions"])[axes_to_be_plotted]
    col_x, col_y = axes_to_be_plotted
    limits_x, limits_y = config_data["partial_lambda_ranges"]
    limits_x[1] = limits_x[1] - (limits_x[1] - limits_x[0]) / dim_x
    limits_y[1] = limits_y[1] - (limits_y[1] - limits_y[0]) / dim_x

    X = data["vertices"][col_x].values.reshape(dim_y, dim_x)
    Y = data["vertices"][col_y].values.reshape(dim_y, dim_x)

    U = data["velocities"][col_x].values.reshape(dim_y, dim_x)
    V = data["velocities"][col_y].values.reshape(dim_y, dim_x)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig, axes = fma.newfig(0.8, ratio=0.8)
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])
    every = 10
    axes.streamplot(X, Y, U, V, density=[2.4, 2.4], linewidth=0.5, arrowsize=0.3)

    # if len(separatrices) > 0:
    #     axes.scatter(separatrices[0][::every], separatrices[1][::every], s=1.0)

    if "explicit_points" in config_data:
        for idx, explicit_point in enumerate(config_data["explicit_points"]["explicit_points"]):
            x, y = np.array(explicit_point)[axes_to_be_plotted]
            axes.scatter(x=x, y=y, s=12.0, label="Expl. point " + str(idx))

    axes.set_xlabel('$' + axes_names[0] + '$')
    axes.set_ylabel('$' + axes_names[1] + '$')

    axes.set_title("Flow field for " + title)
    #
    # x, y, z = config_data["fixed_points"]["fixed_points"][1]
    # axes.scatter(y, z, s=12.0, label="mh2 = 0.187")
    #
    # x, y, z = config_data["fixed_points"]["fixed_points"][2]
    # axes.scatter(y, z, s=12.0, label="mh2 = -0.1645")

    # axes.scatter([-0.226227508697766], [-0.0603059942535251], s=12.0, c="red")

    axes.set_xlim(limits_x)
    axes.set_ylim(limits_y)
    plt.tight_layout()

    # axes.legend()

    fma.savefig(path, "flow_field_" + name)
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("Theory: ", sys.argv[1], "\nPath to project directory: ", sys.argv[2], "\nMode Directory: ", sys.argv[3], "\nRelative path: ", sys.argv[4])
        visualization(theory=sys.argv[1], project_directory=sys.argv[2], mode_directory=sys.argv[3], relative_path=False)
    else:
        # visualization(theory="lorentz_attractor", project_directory="/home/lukas/frgvisualisation/projects/test",
        #               mode_directory="test_vis", relative_path=False)

        visualization(theory="hyperbolic_system", project_directory="/home/lukas/frgvisualisation/projects/FirstProject",
                      mode_directory="visualization", relative_path=False)