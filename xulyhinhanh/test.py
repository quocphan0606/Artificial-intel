'''from PIL import Image

img = Image.open(r"c:\Users\asus\OneDrive\Pictures\Screenshots\z7119309658217_0e0a20c2ec89cd688b1f6ff85a97bbce.jpg").convert('L')   
img.show()//'''
from PIL import Image
import os

filelist = os.listdir()   # lấy danh sách file trong folder

for infile in filelist:
    outfile = os.path.splitext(infile)[0] + ".jpg"

    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print("cannot convert", infile)
'''import os 
from PIL import Image
def get_imlist(path):
    returns a list of filenames for all jpg images in a directory
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('hinhanh1.jpg')]
# khai bao hinh
image = Image.open(' hinhanh1.jpg')
#hinh xam
image = Image.open(' hinhanh1.jpg').convert('L')
# thu nho hinh (lenh thumbnail())
image.thumbnail((128,128))
# cat hinh anh 1 vung nho bang lenh crop() sao chep va dan cac vung
# box =( trai, tren , phai ,duoi)
box =(100,100,400,400)
region = image.crop(box)
region = region.transpose(Image.ROTATE_180)
image.paste(region, box)
# thay doi kich thuoc bang resize()
out = image.resize((128,128))
# xoay hinh anh
out= image.rotate(45)
#--------------------------MATPLOTLIB--------------------------
'''
from PIL import Image
from matplotlib.pylab import *

# read image to array
image = array(Image.open(r"c:\training AI\xulyhinhanh\hinhanh1.jpg"))
axis('off')
# plot the image
imshow(image)

# some points
x = [100,100,400,400]
y = [200,500,200,500]

# plot the points
plot(x,y,'r*')

# line plot connecting the first two points
plot(x[:2],y[:2])

# add title and show the plot
title("plotting image")



'''
plot(x,y) # defound blue solid line
plot(x,y,'r*')# red star_markets
plot(x,y,'go-')# green line with   circle-markets
plot(x,y,'ks:')# black dotted line with square-markets
'''
from PIL import Image
from matplotlib.pylab import *
image = array(Image.open(r"c:\training AI\xulyhinhanh\hinhanh1.jpg").convert('L'))
figure()
# don't use color
gray()
#show coutour with origin 
contour(image,origin='image')
axis('equal')
axis('off')
figure()
hist(image.flatten(),128)


# chú thích tương tác
from PIL import Image
from matplotlib.pylab import *
image = array(Image.open("c:/training AI/xulyhinhanh/hinhanh1.jpg"))
imshow(image)
print('please click 3 point')
x =ginput(3)
print ('you clicked:',x)
show()
#biến đổi cấp độ xám 
from PIL import Image
import numpy as np
from numpy import *
image = array(Image.open('c:/training AI/xulyhinhanh/hinhanh1.jpg').convert('L'))
image2 = 255-image #invert image
image3=(100.0/255)*image+100 #clamp to intervall 100...200
image4=255.0+(image/255.0)**2 #squared
print(int(image.min()), int(image.max()))
# hàm thay đổi kích thước ảnh
def imresize(image,sz):
    '''resize an image aray using PIL'''
    image = Image.fromarray(np.unit8(image))
    return array(image.resize(sz))
