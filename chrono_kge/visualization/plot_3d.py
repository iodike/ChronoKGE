"""
Plot 3D
"""

import matplotlib.pyplot as plt
import pandas as pd

from chrono_kge.knowledge.chrono.timestamp import Timestamp


def plot_3d(x, y, z):
    """"""
    # Dataset
    df = pd.DataFrame({
        'X': x,
        'Y': y,
        'Z': z
    })

    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(df['X'], df['Y'], df['Z'], c='skyblue', s=60)

    for angle in range(0, 360, 2):
        # Turn interactive plotting off
        plt.ioff()

        # Make the plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)

        # Set the angle of the camera
        ax.view_init(30, angle)

        # Save it
        filename = '/Users/ioannisdikeoulias/Downloads/' + str(Timestamp.get_time_in_sec()) + '.png'
        plt.savefig(fname=filename, dpi=96)
        plt.gca()
        plt.close(fig)

    return
