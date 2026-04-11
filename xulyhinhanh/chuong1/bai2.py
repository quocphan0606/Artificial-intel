'''Thực hiện toán tử unsharp bằng cách làm mờ một hình ảnh và sau 
đó trừ ảnh đã mờ vớiảnh gốc. Kết quả cho ra hiệu ứng làm sắc nét cho hình 
ảnh. Hãy thử điều này trên cả hình ảnh màu và thang độ xám'''
'''# gray
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# đọc ảnh grayscale
image = np.array(Image.open('c:/Users/asus/OneDrive/Pictures/hinhnen.jpg').convert('L'))

# làm mờ
blur = gaussian_filter(image, sigma=3)

# mask chi tiết
mask = image - blur

# ảnh sharpen
sharp = image + mask

# hiển thị
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(image,cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(blur,cmap='gray')
plt.title("Blurred")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(sharp,cmap='gray')
plt.title("Sharpened")
plt.axis('off')

plt.show()
'''
# annh mau 3 ket rgb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# đọc ảnh màu
image = np.array(Image.open('c:/Users/asus/OneDrive/Pictures/hinhnen.jpg'))

# làm mờ
blur = gaussian_filter(image, sigma=(3,3,0))

# mask
mask = image - blur

# sharpen
sharp = image + mask

# tránh tràn giá trị
sharp = np.clip(sharp,0,255)

# hiển thị
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(blur)
plt.title("Blurred")

plt.subplot(1,3,3)
plt.imshow(sharp.astype(np.uint8))
plt.title("Sharpened")

plt.show()