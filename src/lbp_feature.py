import numpy as np
from skimage.color import rgb2gray


def get_pixel(image, center, x, y):
    '''
    Compares the center with the chosen pixel and gives a 1 or 0.
    0 if the grayscale value is less then the center pixel grayscale value.
    1 if the grayscale value is grater the the center pixel grayscale value.
    '''
    height, width = image.shape

    if x < 0 or x >= height or y < 0 or y >= width:
        return 0

    if image[x][y] >= center:
        return 1

    return 0


def lbp_calculated_pixel(image, x, y):
    '''
    Calculate the LBP value for a single pixel.
     64 | 128 |   1
    ----------------
     32 |   0 |   2     -from the github
    ----------------
     16 |   8 |   4

    Looks at the 8 pixels around the center pixel, and gives it a value 1 or 0 from the function get_pixel.
    It starts in the top right and then clockwise around.
    Then we get the binary pattern (could look like this: [1, 0, 0, 1, 1, 0, 1, 0]).
    Then we multiply each of thoes 1's and 0's with corresponding powervalue and sum it up.
    This gives us our Local Binary Pattern value for that pixel.
    '''
    center = image[x][y]
    binary_values = []

    binary_values.append(get_pixel(image, center, x - 1, y + 1))
    binary_values.append(get_pixel(image, center, x, y + 1))
    binary_values.append(get_pixel(image, center, x + 1, y + 1))
    binary_values.append(get_pixel(image, center, x + 1, y))
    binary_values.append(get_pixel(image, center, x + 1, y - 1))
    binary_values.append(get_pixel(image, center, x, y - 1))
    binary_values.append(get_pixel(image, center, x - 1, y - 1))
    binary_values.append(get_pixel(image, center, x - 1, y))

    power_values = [1, 2, 4, 8, 16, 32, 64, 128]
    value = 0
    for i in range(len(binary_values)):
        value += binary_values[i] * power_values[i]

    return value


def convert_to_gray(image):
    '''
    Makes the images gray. (who would have thought)
    '''
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        return rgb2gray(image)

    return image


def lbp_image(image):
    '''
    Converts image to gray using convert_to_gray function. 
    Then get the shape of it so we know how many pixels there is in the picture.
    Then we create a list full of zero's for our values, 
    and then we run the lbp_calculated_pixel function for each pixel here (i,j).
    Then saves the Local Binary Pattern value in the list.
    '''
    gray_image = convert_to_gray(image)
    height, width = gray_image.shape
    image_lbp = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            image_lbp[i, j] = lbp_calculated_pixel(gray_image, i, j)

    return image_lbp


def extract_lbp_feature(image):
    '''
    First we get the values with the lbp_image function, and then we flatten it.
    Afterwards we count how many of each value there is.
    Then we divide each of those values with the total sum so we get a normelized data,
    because the pictures can be different sizes.
    '''
    image_lbp = lbp_image(image)

    flat_image = image_lbp.reshape(-1)

    data = np.bincount(flat_image.astype(np.uint8), minlength=256).astype(float) 
    #counts how many of each number there is between 0 to 255, and forces it to be a minimum range of 256 no matter what
    # also makes them floats so we can normelize it.
    data = data / np.sum(data) # normelized it since diffren pictures can have diffrent sizes.

    return data

