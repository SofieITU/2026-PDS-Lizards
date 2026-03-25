import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
data_path = 'data/'

def load_image_and_mask(image_id, data_path=data_path):
    '''
    Docstring for load_image
    
    :param image_id: "img_id" from metadata.csv
    :param data_path: Relative path of the data folder

    This functions takes as input an image ID, 
    and returns the corresponding image and mask 
    (found in "/data/imgs/" and "/data/masks/" respectively)
    as an array
    '''
    
    img_path = data_path + "imgs/"
    mask_path = data_path + "masks/"

    # Load the image/mask
    file_im = img_path + image_id
    file_mask = (mask_path + image_id).replace(".png", "_mask.png")
    im = plt.imread(file_im)
    mask = plt.imread(file_mask)
    
    return im, mask

def extract_features(row, data_path=data_path):

    img_id = row["img_id"]
    im, mask = load_image_and_mask(img_id, data_path) #see how im calling a function i wrote somewhere else
    
    # let's extract the other columns in this row that we're interested in
    diagnostic = row["diagnostic"]

    # using the implementations for asymmetry and compactness defined earlier (this is just for the purpose of the example, this doesn't mean you have to use these. remember you are supposed to understand and improve them)
    asymmetry = get_asymmetry(mask) # quick question: what would happen to this value if there was no mask for the lesion selected?
    compactness = get_compactness(mask)
    color = .....

    # compute your features
    feats = {
        "asymmetry": asymmetry,
        "compactness": compactness, 
        "diagnostic": diagnostic,
        "color": color
    } # 

    return feats

features_df = pd.DataFrame(sampled_dfs.apply(extract_features, axis=1).to_list())

# Colleting all the features together in one dataframe
features_df.to_csv('features.csv', index = False)