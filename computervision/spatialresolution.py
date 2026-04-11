import cv2
import matplotlib.pyplot as plt

# đọc ảnh
img = cv2.imread("e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH02_Original_Images\DIP3E_Original_Images_CH02\Fig0220(a)(chronometer 3692x2812  2pt25 inch 1250 dpi).tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# kích thước ảnh gốc
h, w = img.shape[:2]

# tạo các mức độ phân giải khác nhau
img_300 = cv2.resize(img, (w//2, h//2))   # giảm còn 1/2
img_150 = cv2.resize(img, (w//4, h//4))   # giảm còn 1/4
img_72  = cv2.resize(img, (w//8, h//8))   # giảm còn 1/8

# phóng to lại kích thước ban đầu
img_300_zoom = cv2.resize(img_300, (w, h), interpolation=cv2.INTER_NEAREST)
img_150_zoom = cv2.resize(img_150, (w, h), interpolation=cv2.INTER_NEAREST)
img_72_zoom  = cv2.resize(img_72, (w, h), interpolation=cv2.INTER_NEAREST)

# hiển thị
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(2,2,2)
plt.title("Medium Resolution")
plt.imshow(img_300_zoom)
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Low Resolution")
plt.imshow(img_150_zoom)
plt.axis("off")

plt.subplot(2,2,4)
plt.title("Very Low Resolution")
plt.imshow(img_72_zoom)
plt.axis("off")

plt.show()