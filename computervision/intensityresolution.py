import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH02_Original_Images\DIP3E_Original_Images_CH02\Fig0221(a)(ctskull-256).tif",0)

# giảm xuống 64 mức xám
img_64 = (img//4)*4

# giảm xuống 16 mức xám
img_16 = (img//16)*16

# giảm xuống 4 mức xám
img_4 = (img//64)*64

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Original (256 levels)")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2,2,2)
plt.title("64 levels")
plt.imshow(img_64, cmap='gray')
plt.axis("off")

plt.subplot(2,2,3)
plt.title("16 levels")
plt.imshow(img_16, cmap='gray')
plt.axis("off")

plt.subplot(2,2,4)
plt.title("4 levels")
plt.imshow(img_4, cmap='gray')
plt.axis("off")

plt.show()