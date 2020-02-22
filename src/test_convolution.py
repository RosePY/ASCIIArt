import cv2
from convolution import Convolution as conv
import numpy as np
import time

names = ['i-3-a', 'i-3-b', 'i-3-c']
char = 97

for name in names:
    path = '../input/convolution/' + name + '.jpg'
    img = cv2.imread(path)
    print(path)
    i = 0
    for n in [3, 7, 15]:
        kernel = np.ones((n, n), dtype=np.float32) / (n * n)
        size = str(n) + 'x' + str(n)
        print(size)
        start_time = time.time()
        img_out_1 = conv.convolution_nd(img, kernel, conv_type=conv.CONV_ITERATIVE)
        print("--- classical algorithm: %s seconds ---" % (time.time() - start_time))
        name = '../output/convolution/o-3-' + chr(char) + '-' + str(i) + '-' + size + 'ca.png'
        cv2.imwrite(name, img_out_1)
        i += 1

        start_time = time.time()
        img_out_2 = conv.convolution_nd(img, kernel, conv_type=conv.CONV_KERNEL_SIZE)
        print("--- vectorized element-wise multiplication %s seconds ---" % (time.time() - start_time))
        name = '../output/convolution/o-3-' + chr(char) + '-' + str(i) + '-' + size + 'vem.png'
        cv2.imwrite(name, img_out_2)
        i += 1

        start_time = time.time()
        img_out_3 = conv.convolution_nd(img, kernel, conv_type=conv.CONV_MATRIX_MULT)
        print("--- batch matrix multiplication %s seconds ---" % (time.time() - start_time))
        name = '../output/convolution/o-3-' + chr(char) + '-' + str(i) + '-' + size + 'bmm.png'
        cv2.imwrite(name, img_out_3)
        i += 1

        start_time = time.time()
        img_out_4 = conv.convolution_nd(img, kernel, conv_type=conv.CONV_DYNAMIC_MULT)
        print("--- dynamic element-wise product %s seconds ---" % (time.time() - start_time))
        name = '../output/convolution/o-3-' + chr(char) + '-' + str(i) + '-' + size + 'dem.png'
        cv2.imwrite(name, img_out_4)
        i += 1

        start_time = time.time()
        name = '../output/convolution/o-3-' + chr(char) + '-' + str(i) + '-' + size + 'ocv.png'
        cv2.imwrite(name, cv2.filter2D(img, -1, kernel))
        print("--- opencv %s seconds ---" % (time.time() - start_time))
        i += 1
		
    char += 1
