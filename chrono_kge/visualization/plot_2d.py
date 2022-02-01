"""
Plot 2D
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

from chrono_kge.knowledge.chrono.timestamp import Timestamp

matplotlib.rcParams.update({
    'font.family': 'serif'
})

FOLDER = '/Users/ioannisdikeoulias/Downloads/'


def init():
    """"""
    fig, ax = plt.subplots(dpi=100)
    return fig, ax


def plot_2d(x, y, xlabel="x", ylabel="y", dim=300):
    """"""
    fig, ax = init()

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = dim
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # labels
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
    plt.show()

    # Save it
    filename = FOLDER + xlabel + '-' + ylabel + '-' + str(Timestamp.get_time_in_sec()) + '.png'
    fig.savefig(fname=filename, dpi=300)

    return
