import math


def resample(image, u, v):
    """
    bilinear interpolation of an image.
    :param image:
    :param u: column coordinate
    :param v: row coordinate
    :return: interpolated value of the pixel at position(u, v)
    """
    first_inter = __linear_interpolation(image[:, math.floor(u)], v)
    second_inter = __linear_interpolation(image[:, math.ceil(u)], v)
    return first_inter + (u - math.floor(u)) * (second_inter - first_inter)


def __linear_interpolation(vector, x):
    x_i = math.floor(x)
    x_j = math.ceil(x)

    return vector[x_i] + (x - x_i) * (vector[x_j] - vector[x_i])
