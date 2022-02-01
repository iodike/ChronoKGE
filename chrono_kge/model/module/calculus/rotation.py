"""
2D Rotation
"""

import math
import torch


class Rotation2D:

    def __init__(self, period=1):
        """"""
        self.omega = (2 * math.pi) / period
        return

    def forward(self, x, y, t):
        """"""
        theta = self.omega * t
        xr = x * torch.cos(theta) - y * torch.sin(theta)
        yr = x * torch.sin(theta) + y * torch.cos(theta)
        return xr, yr


class Rotation3D:

    def __init__(self, period=1):
        """"""
        self.omega = (2 * math.pi) / period
        return

    def forward(self, x, y, z, t, axis=0):
        """"""
        theta = self.omega * t

        if axis == 1:
            xr, yr, zr = self.rotate_y(x, y, z, theta)
        elif axis == 2:
            xr, yr, zr = self.rotate_z(x, y, z, theta)
        else:
            xr, yr, zr = self.rotate_x(x, y, z, theta)

        return xr, yr, zr

    @staticmethod
    def rotate_x(x, y, z, theta):
        """"""
        xr = x
        yr = y * torch.cos(theta) - z * torch.sin(theta)
        zr = y * torch.sin(theta) + z * torch.cos(theta)
        return xr, yr, zr

    @staticmethod
    def rotate_y(x, y, z, theta):
        """"""
        xr = x * torch.cos(theta) + z * torch.sin(theta)
        yr = y
        zr = -x * torch.sin(theta) + z * torch.cos(theta)
        return xr, yr, zr

    @staticmethod
    def rotate_z(x, y, z, theta):
        """"""
        xr = x * torch.cos(theta) - y * torch.sin(theta)
        yr = x * torch.sin(theta) + y * torch.cos(theta)
        zr = z
        return xr, yr, zr
