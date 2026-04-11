import cv2 
img_path= "c:/Users/asus/OneDrive/Pictures/hinhnen.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
L= 256
img_neg =L-1-img
cv2.imshow('original image',img)
cv2.imshow('negative image',img_neg)
cv2.waitKey(0)
cv2.destroyAllWindows()