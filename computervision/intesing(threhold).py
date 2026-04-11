import cv2
import numpy as np

img = cv2.imread("e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0312(a)(kidney).tif",0)

h, w = img.shape

slice1 = np.zeros((h,w), np.uint8)
slice2 = img.copy()

mean = np.mean(img)

for x in range(h):
    for y in range(w):

        pixel = img[x,y]

        # method 1
        if 150 <= pixel <= 255:
            slice1[x,y] = 255

        # method 2
        if (mean-35) <= pixel <= (mean+35):
            slice2[x,y] = 0
cv2.imshow("Original", img)
cv2.imshow("Intensity Slicing 1", slice1)
cv2.imshow("Intensity Slicing 2", slice2)

cv2.waitKey(0)
cv2.destroyAllWindows()