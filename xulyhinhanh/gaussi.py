from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
from scipy.ndimage import filters

'''image = array(Image.open('c:/Users/asus/OneDrive/Pictures/hinhnen.jpg').convert('L'))

# Gaussian blur
image2 = filters.gaussian_filter(image,5)'''

im = array(Image.open('c:/Users/asus/OneDrive/Pictures/hinhnen.jpg'))
im2= zeros(im.shape)
for i in range(3):
    im2[:,:,i] = filters.gaussian_filter(im[:,:,i],5)
im2 =array(im2,'unit8')
# hiển thị
plt.figure()

plt.subplot(1,2,1)
plt.imshow(im,cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(im2,cmap='gray')
plt.title("Gaussian sigma=2")
plt.axis('off')

plt.show()