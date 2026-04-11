import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# 1. Tạo một ảnh đơn giản (giống chữ D)
# --------------------------------

img = np.zeros((100,100))

# tạo nền xám
img[20:80,20:80] = 0.5

# tạo chữ D trắng
for i in range(30,70):
    img[i,35] = 1

for j in range(35,60):
    img[30,j] = 1
    img[69,j] = 1

for i in range(30,70):
    img[i,60] = 1

# --------------------------------
# 2. Hiển thị ảnh (Figure 2.18b)
# --------------------------------

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.title("Image Representation f(x,y)")
plt.axis('off')
plt.show()


# --------------------------------
# 3. Surface Plot (Figure 2.18a)
# --------------------------------

x = np.arange(0,img.shape[0])
y = np.arange(0,img.shape[1])

X,Y = np.meshgrid(x,y)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X,Y,img,cmap='gray')

ax.set_title("Surface Representation z = f(x,y)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Intensity")

plt.show()


# --------------------------------
# 4. Matrix Representation (Figure 2.18c)
# --------------------------------

print("Matrix representation f(x,y):")
print(img[25:40,25:40])  # in một phần nhỏ của ảnh