import numpy as np
from PIL import Image
from src.filepath_generator import get_filepath_for


def load_image(filepath):
    '''
    Load an image file from the file path.
    :param filepath: path to the image file (relative path can be used)
    :return: numpy array filled by image intensity values
    '''

    image = Image.open(filepath)
    array = np.array(image.getdata())

    return array


def load_images(start=0, end=2429):
    '''
    Load the range of the images according to the naming convention
    of images (hardcoded at the moment).
    (Note that files are enumerated from 1,
    thus file enumeration is shifted down by one).
    :param start: zero-based range start (included)
    :param end: zero-based range end (excluded)
    :return: 2 dimensional array having one image on each row
    '''
    image_list = np.array([])
    start += 1

    if start < 1:
        start = 1

    if end > 2429:
        end = 2429

    end += 1

    for i in range(start, end):
        filepath = get_filepath_for(i)
        image = load_image(filepath)
        image_list = np.append(image_list, image)

    image_dim = len(image)
    image_list = image_list.reshape((end - start), image_dim)

    return image_list
