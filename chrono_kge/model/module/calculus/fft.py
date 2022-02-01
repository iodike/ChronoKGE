"""
FFT
"""

import numpy as np
import scipy.fft as fft


class FFT:

    def __init__(self, cp=0.2):
        """"""
        self.cp = cp
        return

    def window(self, x):
        """
        Signal filter window
        """
        N = x.shape[0]
        window = np.zeros(N)
        window[:int(N * self.cp)] = 1
        return window

    def fft_compress(self, x, a=0, b=100):
        """"""
        y = fft.fft(x, norm='ortho')
        rx = np.asarray(fft.ifft(y * self.window(x), norm='ortho'))
        rx = np.interp(rx, (np.min(x), np.max(x)), (a, b - 1)).astype(int)
        return rx

    def dct_compress(self, x, a=0, b=100):
        """"""
        y = fft.dct(x, norm='ortho')
        rx = np.asarray(fft.idct(y * self.window(x), norm='ortho'))
        rx = np.interp(rx, (np.min(x), np.max(x)), (a, b - 1)).astype(int)
        return rx

    def dst_compress(self, x, a=0, b=100):
        """"""
        y = fft.dst(x, norm='ortho')
        rx = np.asarray(fft.idst(y * self.window(x), norm='ortho'))
        rx = np.interp(rx, (np.min(x), np.max(x)), (a, b - 1)).astype(int)
        return rx
