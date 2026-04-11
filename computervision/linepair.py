import numpy as np
import matplotlib.pyplot as plt

width = 800
height = 400

img = np.ones((height, width))

Ws = [40,20,10,5,2]  # các mức line pair

start_y = 0
block_height = height//len(Ws)

for W in Ws:
    for i in range(0, width, W*2):
        img[start_y:start_y+block_height, i:i+W] = 0
    start_y += block_height

plt.figure(figsize=(10,5))
plt.imshow(img, cmap='gray')
plt.title("Line Pair Resolution Chart")
plt.axis("off")
plt.show()