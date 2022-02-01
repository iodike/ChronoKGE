"""
Tensor products
"""

import torch


class Products:

    def __init__(self):
        """"""
        return

    @staticmethod
    def khatri_rhao(a, b):
        """
        Khatri-Rhao product (column-wise Kronecker product).
        :param a: [n] x [k]
        :param b: [m] x [k]
        :return: [n*m] x [k]
        """
        return torch.vstack([torch.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T
