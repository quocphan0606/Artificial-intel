import cv2
import numpy as np 
img_path = "e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0305(a)(DFT_no_log).tif"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
c=255/np.log(255+1)
img_log = np.uint8(c*np.log(1.0+img))
cv2.imshow('original image', img)
cv2.imshow('log transformation image',img_log)
print(img[img != 0])
print(img_log[img_log != 0])
cv2.waitKey(0)
cv2.destroyAllWindows()