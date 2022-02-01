"""
Network Graph
"""

import networkx as nx
import matplotlib.pyplot as plt

from datetime import datetime


class NetworkGraph:

    def __init__(self, uid):
        self.G = nx.MultiGraph()
        self.uid = uid
        return

    def add_nodes(self, nodes):
        self.G.add_nodes_from(nodes)
        return

    def add_edges(self, edges):
        self.G.add_weighted_edges_from(edges)
        return

    def draw(self):
        now = datetime.now().strftime('%Y-%m-%d')
        nx.draw(self.G, with_labels=True, font_weight='bold')
        plt.savefig("plots/graph_" + self.uid + "_" + now + ".png")
        return

    def plot(self, triples):
        """

        :return:
        """

        nodes = []
        edges = []

        for triple in triples:
            nodes.append(triple[0])
            nodes.append(triple[2])
            edges.append((triple[0], triple[2], 1))

        self.add_nodes(set(nodes))
        self.add_edges(set(edges))
        self.draw()

        return
