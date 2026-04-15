import numpy as np
from sklearn.cluster import KMeans
from skimage import color
from skimage.segmentation import slic
from skimage.color import rgb2hsv, hsv2rgb
from statistics import variance, stdev
from scipy.stats import circmean, circvar, circstd
from math import sqrt, floor, ceil, nan, pi

# this functions spilts the image in n segments and is used by all of the functions
def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    '''Get color segments of lesion from SLIC algorithm.
    Optional argument n_segments (defualt 50) defines desired amount of segments.
    Optional argument compactness (defualt 0.1) defines balance between color
    and position.

    Args:
        image (numpy.ndarray): image to segment
        mask (numpy.ndarray):  image mask
        n_segments (int, optional): desired amount of segments
        compactness (float, optional): compactness score, decides balance between
            color and and position

    Returns:
        slic_segments (numpy.ndarray): SLIC color segments.
    '''
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)

    return slic_segments




# Computes the variance of mean HSV (Hue, Saturation, Value) colors across all SLIC segments within the lesion.
# hue_var: Variance of hue (color type). you get a value between 0 and 1. 0 means they have the same hue ie. the same color
# 1 means the segments have diffrent hues ie. many colors

# sat_var: variance of saturation (color intensity).0 to dunno yet, but usually not that big.
#  0 mean the same color intencity. higher numbers mean there is both pale and vivid areas

# val_var: Variance of brightness. 0 to dunno, but usually not that big. 0 means the same brightness in the picture.
# higher means both dark and bright segments
# If only one lesion segment exists, the function returns (0, 0, 0).

def hsv_var(image, mask):
    '''Get variance of HSV means for each segment in
    SLIC segmentation in hue, saturation and value channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation

    Returns:
        hue_var (float): variance in hue channel segment means
        sat_var (float): variance in saturation channel segment means
        val_var (float): variance in value channel segment means.
    '''
    slic_segments = slic_segmentation(image, mask)

    if image.shape[-1] == 4:
        image = image[..., :3]

    if len(np.unique(slic_segments)) == 2:
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    n = len(hsv_means) 

    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means:
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    hue_var = circvar(hue, high=1, low=0)
    sat_var = variance(sat, sum(sat)/n)
    val_var = variance(val, sum(val)/n)

    return hue_var, sat_var, val_var




# red_var: variation in how much red there is across the lesion
# green_var: variation in how much green there is across the lesion
# blue_var: variation in how much blue there is across the lesion
# all of them 0 to something. higher numbers mean higher variance in that colot

def rgb_var(image, mask):
    '''Get variance of RGB means for each segment in
    SLIC segmentation in red, green and blue channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation

    Returns:
        red_var (float): variance in red channel segment means
        green_var (float): variance in green channel segment means
        blue_var (float): variance in green channel segment means.
    '''
    slic_segments = slic_segmentation(image, mask)

    if len(np.unique(slic_segments)) == 2:
        return 0, 0, 0

    rgb_means = get_rgb_means(image, slic_segments)
    n = len(rgb_means) 

    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    red_var = variance(red, sum(red)/n)
    green_var = variance(green, sum(green)/n)
    blue_var = variance(blue, sum(blue)/n)

    return red_var, green_var, blue_var


# # If include_ratios = False:
# returns flattened HSV values of the dominant colors.
# With 5 clusters: H1,S1,V1,H2,S2,V2,H3,S3,V3,H4,S4,V4,H5,S5,V5
# [H1, S1, V1] = first dominant color
# [H2, S2, V2] = second dominant color
# and so fourth
# H = color type, S = color intensity, V = brightness.
# All are between 0 and 1.
# # If include_ratios = True:
# returns a list of tuples: (ratio, color)
# returns (ratio, [H,S,V]) for each dominant color.
# ratio tells how much of the lesion that color covers.
def color_dominance(image, mask, clusters = 5, include_ratios = False):
    '''Get the most dominent colors of the cut image that closest sorrounds the lesion using KMeans

    Args:
        image (numpy.ndarray): image to compute dominent colors for
        mask (numpy.ndarray): mask of lesion
        clusters (int, optional): amound of clusters and therefore dominent colors (defualt 3)
        include_ratios (bool, optional): whether to include the domination ratios for each color (defualt False)

    Return:
        if include_ratios == True:
            p_and_c (list): list of tuples, each containing the percentage and RGB array of the dominent color
        else:
            dom_colors (array): array of RGB arrays of each dominent color.
    '''
    if image.shape[-1] == 4:
        image = image[..., :3]

    cut_im = cut_im_by_mask(image, mask) 
    hsv_im = rgb2hsv(cut_im)
    flat_im = np.reshape(hsv_im, (-1, 3))
    k_means = KMeans(n_clusters=clusters, n_init=10, random_state=0)
    k_means.fit(flat_im)

    dom_colors = np.array(k_means.cluster_centers_, dtype='float32')

    if include_ratios:

        counts = np.unique(k_means.labels_, return_counts=True)[1] 
        ratios = counts / flat_im.shape[0] 
        r_and_c = zip(ratios, dom_colors)
        r_and_c = sorted(r_and_c, key=lambda x: x[0],reverse=True) 

        return r_and_c

    flat_dom_colors = dom_colors.flatten()
    return flat_dom_colors


# F1, F2, F3: is the relative RGB composition of the lesion (it sums to 1).
# Tt Indicate how red, green, or blue the lesion is.
# F10, F11, F12: is th difference between the lesion and surrounding skin in the RGB channels.
# Positive meaning lesion has more of that color than skin.
# Negative meaning lesion has less of that color than skin.
def get_relative_rgb_means(image, mask):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''
    if image.shape[-1] == 4:
        image = image[..., :3]

    slic_segments = slic_segmentation(image, mask)

    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(0, max_segment_id + 1):
        segment = image.copy()
        segment[slic_segments != i] = -1

        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))

        rgb_means.append(rgb_mean)

    rgb_means_lesion = np.mean(rgb_means[1:],axis=0)
    rgb_means_skin = np.mean(rgb_means[0])

    F1, F2, F3 = rgb_means_lesion/sum(rgb_means_lesion)
    F10, F11, F12 = rgb_means_lesion - rgb_means_skin

    return F1, F2, F3, F10, F11, F12

# the following 4 functions are helper functions They just work and i have not look at them much yet.
def get_hsv_means(image, slic_segments):
    '''Get mean HSV values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        hsv_means (list): HSV mean values for each segment.
    '''

    hsv_image = rgb2hsv(image)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        segment = hsv_image.copy()
        segment[slic_segments != i] = nan
        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') 
        sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i)) 
        val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) 

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)

    return hsv_means


def get_rgb_means(image, slic_segments):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''

    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(1, max_segment_id + 1):

        segment = image.copy()
        segment[slic_segments != i] = -1

        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))

        rgb_means.append(rgb_mean)

    return rgb_means


def cut_mask(mask):
    '''Cut empty space from mask array such that it has smallest possible dimensions.

    Args:
        mask (numpy.ndarray): mask to cut

    Returns:
        cut_mask_ (numpy.ndarray): cut mask
    '''
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]
    return cut_mask_


def cut_im_by_mask(image, mask):
    '''Cut image array such that it has smallest possible dimensions in accordance with its mask.

    Args:
        image (numpy.ndarray): image to cut
        mask (numpy.ndarray): mask of image to use for cutting

    Returns:
        cut_image (numpy.ndarray): cut image
    '''

    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_image = image[row_min:row_max+1, col_min:col_max+1]

    return cut_image

