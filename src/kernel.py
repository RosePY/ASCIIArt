import numpy as np

class Kernel:

    @staticmethod
    def bilinear_interpolation(n):
        # Parameters: n is the size of kernel rows and columns
        a = np.zeros(n, np.float32)
        for i in range(0, n):
            a[i] = (i + 1) / n
            a[-(i + 1)] = (i + 1) / n
        aa = np.outer(a, a)
        return aa / np.sum(aa)

    @staticmethod
    def gaussian(size = 3, sigma = 1):
        x = np.linspace(- (size // 2), size // 2, size)
        x /= np.sqrt(2) * sigma
        x2 = x ** 2
        kernel = np.exp(- x2[:, None] - x2[None, :])
        return kernel / kernel.sum()