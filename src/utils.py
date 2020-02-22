import numpy as np
import cv2

def load_img_gray(namefile):
    image = cv2.imread(namefile)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

def load_img_np(namefile):
    return cv2.cvtColor(cv2.imread(namefile), cv2.COLOR_BGR2RGB)
