import numpy as np
from border_handling import BorderHandling as bh


class Convolution:
    CONV_ITERATIVE = 1
    CONV_KERNEL_SIZE = 2
    CONV_MATRIX_MULT = 3
    CONV_DYNAMIC_MULT = 4
    CONV_DEFAULT = CONV_KERNEL_SIZE

    @staticmethod
    def convolution(img_in, kernel, conv_type = CONV_DEFAULT, border_type = bh.BORDER_DEFAULT):
        img_with_borders = bh.border_handling(img_in, kernel, border_type)
        img_out = np.empty((img_in.shape), dtype=np.float32)

        if conv_type == Convolution.CONV_ITERATIVE:
            return Convolution.iterative_convolution(img_out, img_with_borders, kernel)
        elif conv_type == Convolution.CONV_KERNEL_SIZE:
            return Convolution.kernel_size_convolution(img_out, img_with_borders, kernel)
        elif conv_type == Convolution.CONV_MATRIX_MULT:
            return Convolution.matrix_mult_convolution(img_out, img_with_borders, kernel)
        elif conv_type == Convolution.CONV_DYNAMIC_MULT:
            return Convolution.dynamic_mult_convolution(img_out, img_with_borders, kernel)

    @staticmethod
    def iterative_convolution(img_out, img_with_borders, kernel):
        rows, cols = img_with_borders.shape
        kernel_rows, kernel_cols = kernel.shape

        for i in range(kernel_rows - 1, rows):
            for j in range(kernel_cols - 1, cols):
                acc = 0
                for n in range(kernel_rows):
                    for m in range(kernel_cols):
                        acc += img_with_borders[i - n][j - m] * kernel[n][m]
                img_out[i - kernel_rows + 1][j - kernel_rows + 1] = acc

        return img_out

    @staticmethod
    def kernel_size_convolution(img_out, img_with_borders, kernel):
        rows, cols = img_with_borders.shape
        kernel_rows, kernel_cols = kernel.shape

        for i in range(rows - kernel_rows + 1):
            for j in range(cols - kernel_cols + 1):
                img_out[i, j] = np.sum(np.multiply(img_with_borders[i:(i + kernel_rows), j:(j + kernel_cols)], kernel))

        return img_out

    @staticmethod
    def matrix_mult_convolution(img_out, img_with_borders, kernel):
        rows, cols = img_with_borders.shape
        kernel_rows, kernel_cols = kernel.shape

        aa = np.empty((((rows - kernel_rows + 1) * (cols - kernel_cols + 1)), (kernel_rows * kernel_cols)), dtype=np.float32)
        bb = kernel.reshape((kernel_rows * kernel_cols, 1))

        for i in range(rows - kernel_rows + 1):
            for j in range(cols - kernel_cols + 1):
                aa[i * (cols - kernel_cols + 1) + j] = img_with_borders[i:(i + kernel_rows), j:(j + kernel_cols)].reshape(kernel_rows * kernel_cols)

        img_out = np.matmul(aa, bb).reshape((rows - kernel_rows + 1, cols - kernel_cols + 1))

        return img_out

    @staticmethod
    def dynamic_mult_convolution(img_out, img_with_borders, kernel):
        rows, cols = img_with_borders.shape
        kernel_rows, kernel_cols = kernel.shape

        for i in range(rows - kernel_rows + 1):
            col_sums = np.zeros((kernel_cols, 1), dtype=np.float32)
            for j in range(cols - kernel_cols + 1):
                if j == 0:
                    col_sums = np.sum(np.multiply(img_with_borders[i:(i + kernel_rows), j:(j + kernel_cols)], kernel), 0)
                else:
                    col_sums = np.delete(col_sums, 0)
                    col_sums = np.append(col_sums, np.dot(img_with_borders[i:(i + kernel_rows), j + kernel_cols - 1], kernel[:, kernel_cols - 1]))
                img_out[i][j] = np.sum(col_sums)
        return img_out

    @staticmethod
    def median_filter2d(img, filter_size):
        # This filter uses as border handling: BORDER_REPLICATE
        rows, cols = img.shape[:2]
        bimg = np.zeros((img.shape[0] + filter_size - 1, img.shape[1] + filter_size - 1), np.float32)
        bimg = bh.border_replicate(bimg, img, filter_size // 2, filter_size // 2)
        new_img = np.zeros((img.shape[0], img.shape[1]), np.float32)

        for i in range(rows):
            for j in range(cols):
                tmp = bimg[i:i + filter_size, j:j + filter_size]
                tmp = tmp.flatten()
                tmp.sort()
                new_img[i][j] = tmp[tmp.size // 2]
        return new_img

    @staticmethod
    def convolution_nd(img, kernel, conv_type = CONV_DEFAULT, border_type = bh.BORDER_DEFAULT):  # convolution for n dimensions
        new_img = np.zeros(img.shape, np.float32)
        if len(img.shape) == 2:
            new_img[:, :] = Convolution.convolution(img[:, :], kernel, conv_type, border_type)
        else:
            for i in range(img.shape[2]):
                new_img[:, :, i] = Convolution.convolution(img[:, :, i], kernel, conv_type, border_type)
        return new_img

    @staticmethod
    def median_filter_nd(img, kernel_size): # median filter for n dimensions
        if len(img.shape) == 2:
            new_img = np.zeros(img.shape)
            new_img[:, :] = Convolution.median_filter2d(img, kernel_size)

        else:
            new_img = np.zeros(img.shape, np.float32)
            for i in range(img.shape[2]):
                new_img[:, :, i] = Convolution.median_filter2d(img[:, :, i], kernel_size)
        return new_img

    @staticmethod
    def threshold(img):
        img[img > 255] = 255
        img[img < 0] = 0
        return img


