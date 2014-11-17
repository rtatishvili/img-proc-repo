import numpy as np

from PIL import Image


def combine_magnitude_and_phase(image_array):
    """
    This method creates grayscale image from array.
    @param image_array: from which will be the image created.
    @return: grayscale image 
    """
    return Image.fromarray(np.uint8(image_array), 'L')


def read_image(image_location, as_array):
    """
    Read image from given location. If as_array == True then the method returns the image as array. 
    @param image_location: path of the image
    @param as_array: if true the method returns the image as array 
    """
    image = Image.open(image_location, 'r')
    if as_array == True:
        data = []
        pix = image.load()
        for y in xrange(image.size[1]):
            data.append([pix[x, y] for x in xrange(image.size[0])])
        return np.array(data)
    return image

def save_image(image_array, location):
    """
    This method save array as JPG image. Client must provide the array and the location where the image will be saved.
    @param image_array: that is saved
    @param location: where the image is saved
    """
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    result = Image.fromarray((image_array * 255).astype(np.uint8))
    result.save(location)
    return result, image_array

