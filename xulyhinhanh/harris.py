from scipy.ndimage import filters
from numpy import *
from matplotlib.pyplot import *
def compute_harris_response(im, sigma=5):
    """
    Compute the Harris corner detector response function
    for each pixel in a grayscale image.
    """

    # derivatives
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0,1), imx)

    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1,0), imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy**2
    Wtr  = Wxx + Wyy

    return Wdet / (Wtr + 1e-12)
# chon lua tot nhat
def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """
    Return corners from a Harris response image.
    min_dist is the minimum number of pixels separating
    corners and image boundary.
    """

    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T

    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # sort candidates
    index = argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []

    for i in index:
        if allowed_locations[coords[i,0], coords[i,1]] == 1:
            filtered_coords.append(coords[i])

            allowed_locations[
                (coords[i,0]-min_dist):(coords[i,0]+min_dist),
                (coords[i,1]-min_dist):(coords[i,1]+min_dist)
            ] = 0

    return filtered_coords

# ham ve diem goc
def plot_harris_points(image, filtered_coords):
    """ Plots corners found in image. """

    figure()
    gray()
    imshow(image)

    plot([p[1] for p in filtered_coords],
         [p[0] for p in filtered_coords],
         '*')

    axis('off')
    show()
def get_descriptors(image, filtered_coords, wid=5):
    """ 
    For each point return pixel values around the point 
    using a neighbourhood of width 2*wid+1.
    (Assume points are extracted with min_distance > wid).
    """

    desc = []

    for coords in filtered_coords:
        patch = image[
            coords[0]-wid : coords[0]+wid+1,
            coords[1]-wid : coords[1]+wid+1
        ].flatten()

        desc.append(patch)

    return desc
def match(desc1, desc2, threshold=0.5):
    """
    For each corner point descriptor in the first image,
    select its match to second image using normalized cross correlation.
    """

    n = len(desc1[0])

    # pair-wise distances
    d = -ones((len(desc1), len(desc2)))

    for i in range(len(desc1)):
        for j in range(len(desc2)):

            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])

            ncc_value = sum(d1 * d2) / (n-1)

            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = argsort(-d)

    matchscores = ndx[:,0]

    return matchscores
