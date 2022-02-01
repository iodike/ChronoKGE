"""
Plot Bar
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "font.size": 18
})


OUTPUT = '/Users/ioannisdikeoulias/Downloads/'

BAR_WIDTH = 0.8

NR14 = 230
NR15 = 251


def init():
    """"""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    return fig, ax


def relation_stats(values1, values2, xticks, xlabel="", ylabel="", name="figure"):
    """"""
    init()

    x1 = [(i*3)+1 for i in range(0, 10)]
    # x2 = [i+1 for i in x1]
    # x3 = [(a+b)/2 for a, b in list(zip(x1, x2))]

    y1 = [(i / NR14) * 100.0 for i in values1]
    y2 = [(i / NR15) * 100.0 for i in values2]

    plt.fill_between(x1, y1, color=(0.3, 0.1, 0.4), alpha=0.8, label='icews14')
    plt.fill_between(x1, y2, color=(0.3, 0.5, 0.4), alpha=0.8, label='icews05-15')

    # plt.bar(x1, y1, width=BAR_WIDTH, color=(0.3, 0.1, 0.4, 0.6), label='icews14')
    # plt.bar(x2, y2, width=BAR_WIDTH, color=(0.3, 0.5, 0.4, 0.6), label='icews05-15')

    plt.xticks(x1, xticks, rotation=45)

    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPUT + name + '.png')

    return


def cae_metrics():
    """"""
    fig, ax = init()

    # simple
    df1 = pd.DataFrame({'x_values': range(100, 600, 100), 'y_values': [
        0.463, 0.499, 0.503, 0.504, 0.505
    ]})

    df2 = pd.DataFrame({'x_values': range(100, 600, 100), 'y_values': [
        0.610, 0.631, 0.631, 0.629, 0.629
    ]})

    df3 = pd.DataFrame({'x_values': range(100, 600, 100), 'y_values': [
        0.742, 0.739, 0.736, 0.735, 0.734
    ]})

    # cae
    df4 = pd.DataFrame({'x_values': range(100, 600, 100), 'y_values': [
        0.447, 0.495, 0.504, 0.509, 0.511
    ]})

    df5 = pd.DataFrame({'x_values': range(100, 600, 100), 'y_values': [
        0.612, 0.646, 0.651, 0.653, 0.654
    ]})

    df6 = pd.DataFrame({'x_values': range(100, 600, 100), 'y_values': [
        0.754, 0.767, 0.764, 0.764, 0.764
    ]})

    # simple
    plt.plot('x_values', 'y_values', data=df1, linestyle='-', marker="o", color="purple", label="Hits@1 (STE)")
    plt.plot('x_values', 'y_values', data=df2, linestyle='-', marker="x", color="purple", label="Hits@3 (STE)")
    plt.plot('x_values', 'y_values', data=df3, linestyle='-', marker="d", color="purple", label="Hits@10 (STE)")

    # cae
    plt.plot('x_values', 'y_values', data=df4, linestyle='-', marker="o", color="green", label="Hits@1 (CTE)")
    plt.plot('x_values', 'y_values', data=df5, linestyle='-', marker="x", color="green", label="Hits@3 (CTE)")
    plt.plot('x_values', 'y_values', data=df6, linestyle='-', marker="d", color="green", label="Hits@10 (CTE)")

    xticks = np.arange(100, 600, 100)

    plt.xticks(xticks, [str(x) for x in xticks])

    plt.legend(
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        bbox_transform=fig.transFigure
    )

    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPUT + 'figure4.png')

    return


def sampling_metrics(y1, y2, y3, y4, name):
    """"""
    fig, ax = init()

    df1 = pd.DataFrame({'x_values': [2**i for i in range(0, 11)], 'y_values': y1})

    df2 = pd.DataFrame({'x_values': [2**i for i in range(0, 11)], 'y_values': y2})

    df3 = pd.DataFrame({'x_values': [2**i for i in range(0, 11)], 'y_values': y3})

    df4 = pd.DataFrame({'x_values': [2**i for i in range(0, 11)], 'y_values': y4})

    color = (0.3, 0.5, 0.8)

    plt.plot('x_values', 'y_values', data=df1, linestyle='-', markersize=5, marker="x", color=color,
             label="ICEWS14")
    plt.plot('x_values', 'y_values', data=df2, linestyle='-', markersize=5, marker="o", color=color,
             label="ICEWS05-15")
    plt.plot('x_values', 'y_values', data=df3, linestyle='-', markersize=5, marker="D", color=color,
             label="WD12K")
    plt.plot('x_values', 'y_values', data=df4, linestyle='-', markersize=5, marker="*", color=color,
             label="GDELT5C")

    xticks = [2**i for i in range(0, 11)]

    ax.set_xscale('log')
    plt.xticks(
        xticks,
        [str(x) for x in xticks]
    )

    plt.yticks([i*1e-1 for i in range(0, 10)])

    ax.legend(
        # loc='upper center',
        bbox_to_anchor=(1.0, 0.15),
        fancybox=True,
        shadow=False,
        ncol=2
    )

    plt.xlabel(xlabel="Sampling rate")
    plt.ylabel(ylabel=name + " (%)")

    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPUT + 'time-sampling-' + name + '.png')

    return


if __name__ == '__main__':
    # facts
    # relation_stats(
    #     [56, 24, 62, 25, 34, 11, 14, 2, 2, 0],
    #     [28, 18, 50, 32, 61, 23, 19, 8, 11, 1],
    #     ["1-5", "6-10", "11-50", "51-100", "101-500", "501-1000",
    #      "1001-5000", "5001-10000", "10001-5000", "50001-100000"],
    #     "facts-per-relation"
    # )
    # subjects
    # relation_stats(
    #     [68, 26, 44, 30, 25, 13, 14, 6, 3, 1],
    #     [37, 21, 37, 37, 32, 47, 14, 14, 9, 2],
    #     ["1-5", "6-10", "11-25", "26-50", "51-100", "101-250",
    #      "251-500", "501-1000", "1001-2500", "2501-5000"],
    #     "subjects-per-relation"
    # )
    # objects
    # relation_stats(
    #     [72, 27, 36, 35, 21, 18, 10, 7, 4, 0],
    #     [39, 26, 38, 35, 34, 38, 15, 14, 10, 2],
    #     ["1-5", "6-10", "11-25", "26-50", "51-100", "101-250",
    #      "251-500", "501-1000", "1001-2500", "2501-5000"],
    #     "objects-per-relation"
    # )
    # times
    # relation_stats(
    #     [85, 55, 26, 18, 11, 3, 3, 6, 6, 18],
    #     [134, 57, 25, 6, 5, 2, 3, 4, 5, 9],
    #     ["1-10", "11-40", "41-80", "81-120", "121-160", "161-200",
    #      "201-240", "241-280", "281-320", "321-365"],
    #     "times-per-relation"
    # )
    # cae
    # cae_metrics()
    #mrr
    sampling_metrics(
        [0.586, 0.608, 0.625, 0.637, 0.647, 0.659, 0.671, 0.683, 0.697, 0.717, 0.717],
        [0.562, 0.604, 0.624, 0.636, 0.645, 0.656, 0.671, 0.690, 0.706, 0.722, 0.742],
        [0.316, 0.337, 0.366, 0.386, 0.410, 0.419, 0.419, 0.437, 0.434, 0.438, 0.441],
        [0.274, 0.291, 0.311, 0.335, 0.370, 0.415, 0.467, 0.533, 0.588, 0.665, 0.665],
        "MRR"
    )
    #h@10
    sampling_metrics(
        [0.717, 0.763, 0.787, 0.796, 0.800, 0.802, 0.805, 0.808, 0.811, 0.817, 0.824],
        [0.735, 0.784, 0.760, 0.757, 0.757, 0.758, 0.761, 0.768, 0.771, 0.785, 0.785],
        [0.498, 0.504, 0.522, 0.528, 0.538, 0.543, 0.545, 0.552, 0.544, 0.546, 0.544],
        [0.436, 0.452, 0.469, 0.491, 0.523, 0.565, 0.615, 0.675, 0.721, 0.785, 0.785],
        "Hits@10"
    )
    #h@3
    sampling_metrics(
        [0.632, 0.652, 0.664, 0.670, 0.678, 0.686, 0.694, 0.703, 0.714, 0.731, 0.731],
        [0.608, 0.653, 0.673, 0.684, 0.691, 0.698, 0.709, 0.721, 0.733, 0.749, 0.759],
        [0.341, 0.363, 0.391, 0.414, 0.428, 0.439, 0.443, 0.455, 0.453, 0.455, 0.456],
        [0.294, 0.312, 0.331, 0.356, 0.391, 0.440, 0.495, 0.564, 0.619, 0.695, 0.695],
        "Hits@3"
    )
    #h@1
    sampling_metrics(
        [0.507, 0.535, 0.555, 0.575, 0.592, 0.607, 0.626, 0.641, 0.659, 0.682, 0.682],
        [0.480, 0.519, 0.538, 0.551, 0.562, 0.579, 0.602, 0.629, 0.651, 0.675, 0.701],
        [0.234, 0.258, 0.292, 0.316, 0.349, 0.359, 0.357, 0.380, 0.380, 0.385, 0.389],
        [0.190, 0.208, 0.229, 0.255, 0.291, 0.337, 0.388, 0.458, 0.518, 0.601, 0.601],
        "Hits@1"
    )
    exit(0)
