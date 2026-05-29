import numpy as np
from skimage.transform import rotate

import numpy as np
from skimage.transform import rotate

import numpy as np
from skimage.transform import rotate

def midpointGroup4(mask):
    
    summed = np.sum(mask, axis=0)
    half_sum = np.sum(summed) / 2
    
    for i, n in enumerate(np.add.accumulate(summed)):
        if n >= half_sum:
            return i
        
    return mask.shape[2] // 2 

def crop(mask):
    mid = midpointGroup4(mask)
    y_nonzero, x_nonzero = np.nonzero(mask)
    
    y_lims = [np.min(y_nonzero), np.max(y_nonzero) + 1]
    x_dist = max(abs(np.min(x_nonzero) - mid), abs(np.max(x_nonzero) - mid))
    x_start = max(0, mid - x_dist)
    x_end = min(mask.shape[1], mid + x_dist + 1)
    
    return mask[y_lims[0]:y_lims[1], x_start:x_end]

def get_asymmetry(mask):
    
    scores = []
    for _ in range(6):
        segment = crop(mask)
        area = np.sum(segment)
        if area == 0:
             scores.append(0.0)
        else:
            xor = np.logical_xor(segment, np.flip(segment))
            union = np.logical_or(segment, np.flip(segment))
            score = np.sum(xor) / np.sum(union) 
            scores.append(score)
        mask = rotate(mask, 30) > 0.5 
        
    return sum(scores) / len(scores)