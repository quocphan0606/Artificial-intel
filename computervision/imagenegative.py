import cv2
img_path = "e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0304(a)(breast_digital_Xray).tif"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
L = 256
img_neg = L-1-img
cv2.imshow('original image',img)
cv2.imshow('negative image',img_neg)
cv2.waitKey(0)
cv2.destroyAllWindows()