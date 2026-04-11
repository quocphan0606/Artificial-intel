'''Chụp một ảnh và áp dụng phương pháp làm mờ ảnh dùng mặt nạ 
Gaussian như trong Hình 1.9. Vẽ các đường viền hình ảnh khi tăng giá trị 
của σ. Điều gì xảy ra và giải thích lý do tại sao?'''


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel

# đọc ảnh và chuyển sang grayscale
image = np.array(Image.open('c:/Users/asus/OneDrive/Pictures/hinhnen.jpg').convert('L'))

# Gaussian blur với các sigma khác nhau
blur2 = gaussian_filter(image,2)
blur5 = gaussian_filter(image,5)
blur10 = gaussian_filter(image,10)

# tính edge bằng Sobel
def edge_detect(img):
    sx = sobel(img,axis=0)
    sy = sobel(img,axis=1)
    return np.hypot(sx,sy)

edge2 = edge_detect(blur2)
edge5 = edge_detect(blur5)
edge10 = edge_detect(blur10)

# hiển thị
plt.figure(figsize=(10,8))

plt.subplot(3,2,1)
plt.imshow(blur2,cmap='gray')
plt.title("Gaussian σ=2")
plt.axis('off')

plt.subplot(3,2,2)
plt.imshow(edge2,cmap='gray')
plt.title("Edge σ=2")
plt.axis('off')

plt.subplot(3,2,3)
plt.imshow(blur5,cmap='gray')
plt.title("Gaussian σ=5")
plt.axis('off')

plt.subplot(3,2,4)
plt.imshow(edge5,cmap='gray')
plt.title("Edge σ=5")
plt.axis('off')

plt.subplot(3,2,5)
plt.imshow(blur10,cmap='gray')
plt.title("Gaussian σ=10")
plt.axis('off')

plt.subplot(3,2,6)
plt.imshow(edge10,cmap='gray')
plt.title("Edge σ=10")
plt.axis('off')

plt.show()