"""
Metrics Plot
"""

import os
import time
import matplotlib.pyplot as plt
import pandas as pd


PLOT_DIR = "../output/plot/"
PLOT_PREFIX = "plot_"
PLOT_FILE_TYPE = ".png"


class MetricsPlot:

    def __init__(self, file, description, xlabel, ylabel, xscale, yscale):

        self.file = file
        self.description = description
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xscale = xscale
        self.yscale = yscale
        self.df = pd.read_csv(self.file, header=0)
        self.yvar = ['h@10', 'h@3', 'h@1', 'mrr']
        self.uid = str(hash(time.time()))
        return

    def plot(self):
        """

        :return:
        """

        fig, ax = plt.subplots()

        # description
        plt.title(self.description)

        # precision
        # ax.yaxis.set_ticks(np.arange(0.4, 0.9, 0.02))
        # ax.xaxis.set_ticks(np.arange(0.0, 1e-12, 1e-1))

        # label
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        # H@10
        ax.plot(self.df[self.xlabel], self.df[self.yvar[0]], label=self.yvar[0].upper())

        # H@3
        ax.plot(self.df[self.xlabel], self.df[self.yvar[1]], label=self.yvar[1].upper())

        # H@1
        ax.plot(self.df[self.xlabel], self.df[self.yvar[2]], label=self.yvar[2].upper())

        # MRR
        ax.plot(self.df[self.xlabel], self.df[self.yvar[3]], label=self.yvar[3].upper())

        # legend
        ax.legend(loc=1)

        # scale
        plt.xscale(self.xscale)
        plt.yscale(self.yscale)

        # effects
        plt.grid()
        # plt.show()

        os.makedirs(PLOT_DIR)

        fig.savefig(PLOT_DIR + PLOT_PREFIX + self.uid + PLOT_FILE_TYPE)

        return
