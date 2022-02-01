"""
History
"""

import numpy as np


class History:

    def __init__(self, data, index):
        """"""
        self.data = np.asarray(data)
        self.index = index
        return

    def sort_by_index(self, index):
        """"""
        self.data = self.data[self.data[:, index].argsort()]
        return

    def get_history(self, k):
        """"""
        self.sort_by_index(self.index)

        h = []
        b = self.data[:, self.index][0] + k

        for i, e in enumerate(self.data):

            if e[self.index] < b:
                h.append(e)
            else:
                yield h
                h.clear()
                h.append(e)
                b += k

        yield h
