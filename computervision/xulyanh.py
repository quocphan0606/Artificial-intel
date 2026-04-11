import cv2 
import numpy as np 
img_path = "e:\computervision\Origin-Images-of-Digital-Image-Process-master\Origin-Images-of-Digital-Image-Process-master\Digital_Image_Processing_3rd\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0310(b)(washed_out_pollen_image).tif"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
rmin= np.min(img)
rmax = np.max(img)
rmean = np.mean(img)
def ContrastStretching(img,r1,r2,s1,s2):
    W, H = img.shape
    img_contrast= np.zeros((W,H),np.uint8)
    for x in range(0,W):
        for y in range(0,H):
            r = img[x,y]
            if(r<=r1):
                s=(s1/r1)*r
            elif ((r>r1)&(r<=r2)):
                s=((s2-s1)/(r2-r1))*r+(s1*r2-s2*r1)/(r2-r1)
            else:
                s=(255-r2)/(255-r2)*r+(255*(s2-r2))/(255-r2)
            img_contrast[x,y] = np.uint8(s)
    return img_contrast
img_contrast= ContrastStretching(img,rmin,rmax,0.0,255.0)
img_threshodld= ContrastStretching(img,rmean,rmean,0.0,255.0)
cv2.imshow('orrigan',img)
cv2.imshow("contrast",img_contrast)
cv2.imshow("threshold", img_threshodld)
def GrammaTransform(img, gramma):
    W,H = img.shape
    img_gramma = np.zeros((W, H), np.uint8)
    c= np.power(255,1-gramma)
    for x in range(0, W):
        for y in range(0, H):
            r = img[x,y]
            s=c*np.power(r,gramma)
            img_gramma[x,y] = np.uint8(s)
    return img_gramma

'''img_gramma1 = GrammaTransform(img,3.0)
img_gramma2 = GrammaTransform(img,4.0)
img_gramma3 = GrammaTransform(img,5.0)
cv2.imshow("original image", img)
cv2.imshow("gramma transformation image (gramma = 3)", img_gramma1)
cv2.imshow("gramma transformation image (gramma = 4)", img_gramma2)
cv2.imshow("gramma transformation image (gramma = 5)", img_gramma3)'''
cv2.waitKey(0)
cv2.destroyAllWindows()