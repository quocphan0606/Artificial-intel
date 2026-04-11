import sys
sys.path.append(r"C:\training AI\xulyhinhanh")

import rof
from numpy import *
from numpy import random
from scipy.ndimage import filters



# create synthetic image with noise
image = zeros((500, 500))
image[100:400, 100:400] = 128
image[200:300, 200:300] = 255

image = image + 30 * random.standard_normal((500, 500))

U, T = rof.denoise(image, image)
G = filters.gaussian_filter(image, 10)

# save the result
import scipy.misc
scipy.misc.imsave('synth_rof.pdf', U)
scipy.misc.imsave('synth_gaussian.pdf', G)