"""
Plot Scatter
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns

from chrono_kge.knowledge.chrono.timestamp import Timestamp

matplotlib.rcParams.update({
    'font.family': 'serif'
})


FOLDER = '/Users/ioannisdikeoulias/Downloads/'


def init():
    """"""
    fig, ax = plt.subplots(dpi=100)
    return fig, ax


def plot_scatter(groups: dict):
    """"""

    dfs = []
    for i, group in groups.items():
        group = np.array(group)
        x = group[:, 0]
        y = group[:, 1]
        df = pd.DataFrame({'x': x, 'y': y, 'group': np.repeat(str(i), len(group))})
        dfs.append(df)

    for df in dfs:
        plt.plot('x', 'y', "", data=df, linestyle='', marker='o', markersize=0.5, color=tuple(np.random.rand(3)))

    plt.xlabel('Time')
    plt.ylabel('Subject')
    plt.title('Title', loc='left')
    plt.show()

    return


def plot_bubble(x, y, z):
    """"""
    fig, ax = init()

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    # sns.scatterplot(data=df, x="x", y="y", size="z", legend=False, sizes=(10, 1000))

    plt.xlabel('Time')
    plt.ylabel('#Relation')
    plt.show()

    filename = FOLDER + "time" + '-' + "rcount" + '-' + str(Timestamp.get_time_in_sec()) + '.png'
    fig.savefig(fname=filename, dpi=300)

    return
