import numpy as np
import cv2
from skimage import color, morphology, filters, util

def remove_pen_marks(image, lesion_mask):
    img = util.img_as_float(image)
    lesion_mask = lesion_mask.astype(bool)

    # 1) HSV for colored ink
    hsv = color.rgb2hsv(img)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    colored_pen_mask = ((hue > 0.4) & (hue < 0.9) & (sat > 0.25))

    # 2) Dark ink detection
    gray = color.rgb2gray(img)
    dark_strokes = morphology.black_tophat(gray, footprint=morphology.disk(11))
    thresh = filters.threshold_otsu(dark_strokes)
    dark_pen_mask = dark_strokes > thresh
    dark_pen_mask &= (val < 0.7)

    # 3) Combine
    pen_mask = colored_pen_mask | dark_pen_mask

    # 4) Exclude lesion region
    protected_lesion = morphology.binary_dilation(lesion_mask, morphology.disk(10))
    pen_mask = pen_mask & (~protected_lesion)

    # 5) Clean mask
    pen_mask = morphology.binary_closing(pen_mask, morphology.disk(3))
    pen_mask = morphology.remove_small_objects(pen_mask, min_size=80)
    pen_mask = morphology.remove_small_holes(pen_mask, area_threshold=80)
    pen_mask = morphology.binary_dilation(pen_mask, morphology.disk(5))

    # 6) OpenCV inpainting
    img8 = (img * 255).astype(np.uint8)
    mask8 = (pen_mask.astype(np.uint8)) * 255
    clean_image = cv2.inpaint(img8, mask8, 7, cv2.INPAINT_TELEA)

    return clean_image, pen_mask