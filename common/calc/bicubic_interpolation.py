import math


def resample(image, u, v):
    """
    Bicubic interpolation of an image.
    :param image:
    :param u: column coordinate
    :param v: row coordinate
    :return: interpolated value of the pixel at position(u, v)
    """
    first_inter = __cubic_interpolation(image[:, math.floor(u)-1], v)
    second_inter = __cubic_interpolation(image[:, math.floor(u)], v)
    third_inter = __cubic_interpolation(image[:, math.ceil(u)], v)
    return __cubic_interpolation1(first_inter, second_inter, third_inter, u)


def __cubic_interpolation(vector, x):
    x0 = math.floor(x)
    x1 = math.ceil(x)
    m = 1.5 * (vector[x0-1] - 2*vector[x0] + vector[x1])
    a = -m / 6.
    b = m / 2.
    c = vector[x1] - vector[x0-1] - m/3.
    d = vector[x0]
    return a*(x-x0)**3 + b*(x-x0)**2 + c*(x-x0) + d


def __cubic_interpolation1(first, second, third, x):
    x0 = math.floor(x)
    m = 1.5 * (first - 2*second + third)
    a = -m / 6.
    b = m / 2.
    c = second - first - m/3.
    d = first
    return a*(x-x0)**3 + b*(x-x0)**2 + c*(x-x0) + d