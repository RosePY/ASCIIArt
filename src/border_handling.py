import numpy as np

class BorderHandling:

    BORDER_REPLICATE = 1
    BORDER_REFLECT = 2
    BORDER_REFLECT_101 = 3
    BORDER_WRAP = 4
    BORDER_CONSTANT = 5
    BORDER_DEFAULT = BORDER_REFLECT_101

    @staticmethod
    def border_handling(img_in, kernel, border_type=BORDER_DEFAULT):
        rows, cols = img_in.shape[:2]
        kernel_rows, kernel_cols = kernel.shape
        kernel_center_row, kernel_center_col = int(kernel_rows / 2), int(kernel_cols / 2)
        img_out = np.zeros((kernel_rows + rows - 1, kernel_cols + cols - 1))
        if border_type == BorderHandling.BORDER_REPLICATE:
            return BorderHandling.border_replicate(img_out, img_in, kernel_center_row, kernel_center_col)
        elif border_type == BorderHandling.BORDER_REFLECT:
            return BorderHandling.border_reflect(img_out, img_in, kernel_center_row, kernel_center_col)
        elif border_type == BorderHandling.BORDER_REFLECT_101:
            return BorderHandling.border_reflect_101(img_out, img_in, kernel_center_row, kernel_center_col)
        elif border_type == BorderHandling.BORDER_WRAP:
            return BorderHandling.border_wrap(img_out, img_in, kernel_center_row, kernel_center_col)
        elif border_type == BorderHandling.BORDER_CONSTANT:
            return BorderHandling.border_constant(img_in, kernel_cols, kernel_rows, kernel_cols, kernel_center_row, kernel_center_col, 0)

    @staticmethod
    def border_replicate(img_out, img_in, kernel_center_row, kernel_center_col):
        rows, cols = img_in.shape[:2]
        # corner squares
        img_out[:kernel_center_row, :kernel_center_col] = img_in[0, 0] * np.ones((kernel_center_row, kernel_center_col), dtype=np.float32)
        img_out[:kernel_center_row, -kernel_center_col:] = img_in[0, (cols - 1)] * np.ones((kernel_center_row, kernel_center_col), dtype=np.float32)
        img_out[-kernel_center_row:, :kernel_center_col] = img_in[(rows - 1), 0] * np.ones((kernel_center_row, kernel_center_col), dtype=np.float32)
        img_out[-kernel_center_row:, -kernel_center_col:] = img_in[(rows - 1), (cols - 1)] * np.ones((kernel_center_row, kernel_center_col), dtype=np.float32)
        # sides
        img_out[:kernel_center_row, kernel_center_col:-kernel_center_col] = np.array([img_in[0, :], ] * kernel_center_row)
        img_out[kernel_center_row:-kernel_center_row, :kernel_center_col] = np.array([img_in[:, 0], ] * kernel_center_col).transpose()
        img_out[kernel_center_row:-kernel_center_row, -kernel_center_col:] = np.array([img_in[:, (cols - 1)], ] * kernel_center_col).transpose()
        img_out[-kernel_center_row:, kernel_center_col:-kernel_center_col] = np.array([img_in[(rows - 1), :], ] * kernel_center_row)
        # original image
        img_out[kernel_center_row:-kernel_center_row, kernel_center_col:-kernel_center_col] = img_in
        return img_out

    @staticmethod
    def border_reflect(img_out, img_in, kernel_center_row, kernel_center_col):
        # corner squares
        img_out[:kernel_center_row, :kernel_center_col] = np.rot90(img_in[:kernel_center_row, :kernel_center_col], 2)
        img_out[:kernel_center_row, -kernel_center_col:] = np.rot90(img_in[:kernel_center_row, -kernel_center_col:], 2)
        img_out[-kernel_center_row:, :kernel_center_col] = np.rot90(img_in[-kernel_center_row:, :kernel_center_col], 2)
        img_out[-kernel_center_row:, -kernel_center_col:] = np.rot90(img_in[-kernel_center_row:, -kernel_center_col:], 2)
        # sides
        img_out[:kernel_center_row, kernel_center_col:-kernel_center_col] = np.flip(img_in[:kernel_center_row, :], 0)
        img_out[kernel_center_row:-kernel_center_row, :kernel_center_col] = np.flip(img_in[:, :kernel_center_col], 1)
        img_out[kernel_center_row:-kernel_center_row, -kernel_center_col:] = np.flip(img_in[:, -kernel_center_col:], 1)
        img_out[-kernel_center_row:, kernel_center_col:-kernel_center_col] = np.flip(img_in[-kernel_center_row:, :], 0)
        # original image
        img_out[kernel_center_row:-kernel_center_row, kernel_center_col:-kernel_center_col] = img_in
        return img_out

    @staticmethod
    def border_reflect_101(img_out, img_in, kernel_center_row, kernel_center_col):
        # corner squares
        img_out[:kernel_center_row, :kernel_center_col] = np.rot90(img_in[1:(kernel_center_row + 1), 1:(kernel_center_col + 1)], 2)
        img_out[:kernel_center_row, -kernel_center_col:] = np.rot90(img_in[1:(kernel_center_row + 1), -(kernel_center_col + 1):-1], 2)
        img_out[-kernel_center_row:, :kernel_center_col] = np.rot90(img_in[-(kernel_center_row + 1):-1, 1:(kernel_center_col + 1)], 2)
        img_out[-kernel_center_row:, -kernel_center_col:] = np.rot90(img_in[-(kernel_center_row + 1):-1, -(kernel_center_col + 1):-1], 2)
        # sides
        img_out[:kernel_center_row, kernel_center_col:-kernel_center_col] = np.flip(img_in[1:(kernel_center_row + 1), :], 0)
        img_out[kernel_center_row:-kernel_center_row, :kernel_center_col] = np.flip(img_in[:, 1:(kernel_center_col + 1)], 1)
        img_out[kernel_center_row:-kernel_center_row, -kernel_center_col:] = np.flip(img_in[:, -(kernel_center_col + 1):-1], 1)
        img_out[-kernel_center_row:, kernel_center_col:-kernel_center_col] = np.flip(img_in[-(kernel_center_row + 1):-1, :], 0)
        # original image
        img_out[kernel_center_row:-kernel_center_row, kernel_center_col:-kernel_center_col] = img_in
        return img_out

    @staticmethod
    def border_wrap(img_out, img_in, kernel_center_row, kernel_center_col):
        # corner squares
        img_out[:kernel_center_row, :kernel_center_col] = img_in[-kernel_center_row:, -kernel_center_col:]
        img_out[:kernel_center_row, -kernel_center_col:] = img_in[-kernel_center_row:, :kernel_center_col]
        img_out[-kernel_center_row:, :kernel_center_col] = img_in[:kernel_center_row, -kernel_center_col:]
        img_out[-kernel_center_row:, -kernel_center_col:] = img_in[:kernel_center_row, :kernel_center_col]
        # sides
        img_out[:kernel_center_row, kernel_center_col:-kernel_center_col] = img_in[-kernel_center_row:, :]
        img_out[kernel_center_row:-kernel_center_row, :kernel_center_col] = img_in[:, -kernel_center_col:]
        img_out[kernel_center_row:-kernel_center_row, -kernel_center_col:] = img_in[:, :kernel_center_col]
        img_out[-kernel_center_row:, kernel_center_col:-kernel_center_col] = img_in[:kernel_center_row, :]
        # original image
        img_out[kernel_center_row:-kernel_center_row, kernel_center_col:-kernel_center_col] = img_in
        return img_out

    @staticmethod
    def border_constant(img_in, kernel_rows, kernel_cols, kernel_center_row, kernel_center_col, const):
        rows, cols = img_in.shape[:2]
        img_out = const * np.ones((kernel_rows + rows - 1, kernel_cols + cols - 1), dtype=np.float32)
        img_out[kernel_center_row:-kernel_center_row, kernel_center_col:-kernel_center_col] = img_in
        return img_out