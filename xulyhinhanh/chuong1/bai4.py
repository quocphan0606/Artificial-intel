from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from skimage.transform import hough_line, hough_line_peaks

# đọc ảnh
image = np.array(Image.open('c:/Users/asus/OneDrive/Pictures/hinhnen.jpg').convert('L'))

# gradient
gx = sobel(image, axis=0)
gy = sobel(image, axis=1)

magnitude = np.sqrt(gx**2 + gy**2)

# threshold để lấy edge
edges = magnitude > 50

# Hough transform
h, theta, d = hough_line(edges)

# tìm các đường thẳng mạnh nhất
accum, angles, dists = hough_line_peaks(h, theta, d)

# vẽ kết quả
plt.imshow(image, cmap='gray')

for angle, dist in zip(angles, dists):
    y0 = (dist - 0*np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1]*np.cos(angle)) / np.sin(angle)
    plt.plot((0, image.shape[1]), (y0, y1), '-r')

plt.title("Detected Lines")
plt.axis('off')
plt.show()