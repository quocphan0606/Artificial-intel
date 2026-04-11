from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def gradient_edge(image):

    # gradient theo x
    gx = sobel(image, axis=0)

    # gradient theo y
    gy = sobel(image, axis=1)

    # độ lớn gradient
    grad = np.sqrt(gx**2 + gy**2)

    # threshold
    edge = grad > 50

    return edge