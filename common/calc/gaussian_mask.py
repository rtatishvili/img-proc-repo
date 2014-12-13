
import numpy as np


def __calculate_sigma(m):
    return (m - 1.0) / (2.0 * 2.575)


def __calculate_gauss_dimension(size):
    m = (size - 1.) / 2.
    x = np.arange(-m, m + 1)
    return x * x


def __generate_gauss(size=(3, 3), sigma=0.5):
    p = __calculate_gauss_dimension(size)
    mask = np.exp(-p / (2. * sigma * sigma))
    sum_ = mask.sum()
    if sum_ != 0:
        mask /= sum_
    return mask


def generate_gauss_1d(size=3, y_direction=False, sigma=None):
    """


    :rtype : object
    :type y_direction: object
    :param size:
    :param sigma:
    :return:
    """
    if sigma is None:
        sigma = __calculate_sigma(size)
    if y_direction:
        return __generate_gauss(size, sigma).reshape((size, 1))
    return __generate_gauss(size, sigma)


def generate_gauss_2d(size=(3, 3), sigma=None):
    """
    2D gaussian mask 
    :param size:
    :param sigma:
    """
    if sigma is None:
        sigma = __calculate_sigma(size[0])
    x_dimension = generate_gauss_1d(size[0], y_direction=False, sigma=sigma)
    y_dimension = generate_gauss_1d(size[1], y_direction=True, sigma=sigma)
    return x_dimension*y_dimension