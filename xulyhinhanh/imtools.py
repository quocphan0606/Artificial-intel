import os 
from PIL import Image
from numpy import *
def get_imlist(path):
    '''returns a list of filenames for all jpg images in a directory'''
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('hinhanh1.jpg')]
import numpy as np
def imresize(image,sz):
    '''resize an image aray using PIL'''
    image = Image.fromarray(np.unit8(image))
    return array(image.resize(sz))
def histeq(image,nbr_bins=256):
    '''histogram equalization of a grayscale image'''
    #get image histogram
    imagehist,bin=histogram(image.flatten(),nbr_bins,normed =True)
    cdf =imagehist.cumsum() #cumulative disstribution function
    cdf = 255*cdf/cdf[-1]# normalize
    #use linear interpolation of cdf to find new pixel values
    image2 = interp(image.flatten(),bin[:-1],cdf)
    return image2.reshape(image.shape),cdf
def compute_average(imlist):
    '''compute the average of a list of images'''
    #open first image and make in to arrat of type float
    averageimage = array(Image.open(imlist[0]),'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname +'.....skipped')
    averageim /= len(imlist)
    #return average as unit8
    return(array(averageim,'unit8'))
    
