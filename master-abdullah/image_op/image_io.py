import numpy as np

from PIL import Image


def read_image(image_location, as_array):
    """
    Read image from given location. If as_array == True then the method returns the image as array.
    If as_array == False then the method returns an Image object from PIL
    @param image_location: path of the image
    @param as_array: if true the method returns the image as array, if not then it returns an Image object from PIL
    """
    image = Image.open(image_location, 'r')
    if as_array:
        data = []
        pix = image.load()
        for y in xrange(image.size[1]):
            data.append([pix[x, y] for x in xrange(image.size[0])])
        return np.array(data)
    return image


def normalize_image(array):
    array = 255 * ((array - array.min()) / (array.max() - array.min()))
    return array


def save_array_as_gray_image(array, location, normalize=False):
    """
    Save array as JPG image. Client must provide the array and the location where the image will be saved.
    @:param array: that is saved
    @:param location: where the image is saved, the directory must exist
    @:param normalize: boolean to determine if the array is to be normalized to range 0..255
    """
    assert len(array.shape) == 2, "only 2D arrays are accepted to generate a gray image"

    # normalize array data to range 0..255
    if normalize:
        array = normalize_image(array)

    # convert array to Image object
    result = Image.fromarray(array.astype(np.uint8), 'L')
    result.save(location)

