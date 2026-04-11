import cv2
import matplotlib.pyplot as plt

# đọc ảnh gốc
img = cv2.imread("e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH02_Original_Images\DIP3E_Original_Images_CH02\Fig0220(a)(chronometer 3692x2812  2pt25 inch 1250 dpi).tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# kích thước ảnh gốc
h, w = img.shape[:2]

# shrink ảnh xuống (giả lập 72 dpi)
small_nearest = cv2.resize(img, (213,162), interpolation=cv2.INTER_NEAREST)
small_bilinear = cv2.resize(img, (213,162), interpolation=cv2.INTER_LINEAR)
small_bicubic = cv2.resize(img, (213,162), interpolation=cv2.INTER_CUBIC)

# zoom lại kích thước ban đầu
zoom_nearest = cv2.resize(small_nearest, (w,h), interpolation=cv2.INTER_NEAREST)
zoom_bilinear = cv2.resize(small_bilinear, (w,h), interpolation=cv2.INTER_LINEAR)
zoom_bicubic = cv2.resize(small_bicubic, (w,h), interpolation=cv2.INTER_CUBIC)

# hiển thị
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(2,2,2)
plt.title("Nearest Neighbor")
plt.imshow(zoom_nearest)
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Bilinear Interpolation")
plt.imshow(zoom_bilinear)
plt.axis("off")

plt.subplot(2,2,4)
plt.title("Bicubic Interpolation")
plt.imshow(zoom_bicubic)
plt.axis("off")

plt.show()