import cv2
import numpy as np
import matplotlib.pyplot as plt
img_paths = [
    "e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0316(4)(bottom_left).tif",
    "e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0320(1)(top_left).tif",
    "e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0320(2)(2nd_from_top).tif",
    
    "e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0320(3)(third_from_top).tif"
]
image =" "

images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in img_paths]
histogram = [np.zeros(256) for _ in range(len(img_paths))]

def HistogramCalculate(img):
    W, H = img.shape
    histogram = np.zeros(256)
    
    for x in range(W):
        for y in range(H):
            r = img[x, y]
            histogram[r] += 1
    
    return histogram

def DrawHistogram(histogram):
    plt.bar(range(256), histogram, color='gray')
    plt.xlim([0, 255])

plt.figure("Histogram", figsize=(12, 6))

for i, img in enumerate(images):

    histogram[i] = HistogramCalculate(img)

    plt.subplot(2, 4, 2 * i + 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 4, 2 * i + 2)
    DrawHistogram(histogram[i])
    plt.xticks([])
    plt.yticks([])
def HistogramEqualization(img):

    W, H = img.shape
    hr = HistogramCalculate(img)

    pr = np.zeros(256)
    for i in range(256):
        pr[i] = hr[i] / (W * H)

    hr_eq = np.zeros(256)
    for i in range(256):
        for j in range(i):
            hr_eq[i] = hr_eq[i] + pr[j]

    img_eq = np.zeros((W, H), np.uint8)

    for x in range(W):
        for y in range(H):
            r = img[x, y]
            img_eq[x, y] = 255 * hr_eq[r]

    return img_eq
def HistogramEqualization(img):
    W, H = img.shape
    hr = np.bincount(img.flatten(), minlength=256)
    pr = hr/(W*H)
    hr_eq = np.cumsum(pr)

    img_eq = np.zeros((W,H), np.uint8)
    for x in range(W):
        for y in range(H):
            r = img[x, y]
            img_eq[x, y] = np.uint8(255*hr_eq[r])

    return img_eq


global_histeq_img = HistogramEqualization(image)

a = 3
b = a // 2
W, H = image.shape
local_histeq_img = image.copy()

for x in range(b, W - b):
    for y in range(b, H - b):
        w = image[x-b:x+b+1, y-b:y+b+1]
        w_eq = HistogramEqualization(w)
        local_histeq_img[x, y] = w_eq[b, b]
plt.tight_layout()
plt.show()