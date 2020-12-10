import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

plt.style.use('seaborn-dark-palette')

from figure_management.loading_figure_mode import loading_figure_mode
fma = loading_figure_mode("saving")


def performance_plot1():
    cwd = os.getcwd()
    data = pd.read_csv(cwd + "/../../doc/figures/IdentityCPURunTimes.txt", delimiter=" ", header=None)
    data.index.name = "CPU"

    data2 = pd.read_csv(cwd + "/../../doc/figures/IdentityGPURunTimes.txt", delimiter=" ", header=None)
    data2.index.name = "GPU"

    fig, ax = fma.newfig(0.84)

    ax.plot(data[0], data[1], label="CPU")
    ax.plot(data2[0], data2[1], label="GPU")

    ax.set_xlabel("Dimension $D$ [-]")
    ax.set_ylabel("Run time [s]")
    ax.set_yscale("Log")

    ax.legend()

    plt.grid(True)

    plt.tight_layout()

    fma.savefig(".", "performance_plot1")
    plt.close()


def performance_plot2():

    cwd = os.getcwd()
    data = pd.read_csv(cwd + "/../../doc/figures/ThreePointCPURunTimes.txt", delimiter=" ", header=None)
    data.index.name = "CPU"

    data2 = pd.read_csv(cwd + "/../../doc/figures/ThreePointGPURunTimes.txt", delimiter=" ", header=None)
    data2.index.name = "GPU"

    fig, ax = fma.newfig(0.84)

    ax.plot(data[0], data[1], label="CPU")
    ax.plot(data2[0], data2[1], label="GPU")

    ax.set_xlabel("Number of hypercubes $N_{\\textnormal{hc}} [-]")
    ax.set_ylabel("Run time [s]")
    ax.set_xscale("Log")
    ax.set_yscale("Log")
    ax.set_ylim(1, 100)

    ax.legend()

    plt.grid(True)

    plt.tight_layout()

    fma.savefig(".", "performance_plot2")
    plt.close()


if __name__ == '__main__':
    performance_plot1()
    performance_plot2()
