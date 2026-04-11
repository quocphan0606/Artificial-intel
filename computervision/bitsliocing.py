import cv2
import numpy as np

img_path = "e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0314(a)(100-dollars).tif"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

W, H = img.shape
bit_plane = [np.zeros((W, H), np.uint8) for _ in range(11)]

for x in range(W):
    for y in range(H):
        r = img[x, y]
        for z in range(8):
            mask = (1 << z)
            bit_value = (r & mask) >> z
            bit_plane[z][x, y] = bit_value * 255

bit_plane[8] = ((bit_plane[7] / 255) * 2**7 + (bit_plane[6] / 255) * 2**6).astype(np.uint8)
bit_plane[9] = (bit_plane[8] + (bit_plane[5] / 255) * 2**5).astype(np.uint8)
bit_plane[10] = (bit_plane[9] + (bit_plane[4] / 255) * 2**4).astype(np.uint8)

cv2.imshow("Original image", img)

for z in range(8):
    cv2.imshow(f"Bit plane {z}", bit_plane[z])

cv2.imshow("Bit plane 7+6", bit_plane[8])
cv2.imshow("Bit plane 7+6+5", bit_plane[9])
cv2.imshow("Bit plane 7+6+5+4", bit_plane[10])

cv2.waitKey(0)
cv2.destroyAllWindows()