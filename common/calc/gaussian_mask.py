import numpy as np
import image_op.image_manip as img_mp


def __calculate_sigma(m):
    return (m - 1.0) / (2.0 * 2.575)


def __calculate_distance_from_center(size):
    m = (size - 1.) / 2.
    x = np.arange(-m, m + 1)
    return x


def __generate_gauss(size=(3, 3), sigma=0.5):
    distance = __calculate_distance_from_center(size)**2
    mask = np.exp(-distance / (2. * sigma * sigma))
    sum_ = mask.sum()
    if sum_ != 0:
        mask /= sum_
    return mask


def generate_gauss_1d(size=3, y_direction=False, sigma=None):
    """
    Generates one dimensional Gaussian filter. Client can specify the size of the filter,
    is it transposed and size of the variance.

    :rtype : ndarray
    :type y_direction: True if the vector is transposed
    :param size: of the filter
    :param sigma: size of the variance
    :return: one dimensional Gaussian filter
    """
    if sigma is None:
        sigma = __calculate_sigma(size)
    if y_direction:
        return __generate_gauss(size, sigma).reshape((size, 1))
    return __generate_gauss(size, sigma)


def generate_gauss_2d(size=(3, 3), sigma=None):
    """
    Generates two dimensional Gaussian filter. Client can specify the size of the filter,
    size of the variance.

    :param size: of the mask
    :param sigma: size of the variance
    """
    if sigma is None:
        sigma = __calculate_sigma(size[0])
    x_dimension = generate_gauss_1d(size[0], y_direction=False, sigma=sigma)
    y_dimension = generate_gauss_1d(size[1], y_direction=True, sigma=sigma)
    return x_dimension * y_dimension


def zero_pad_mask(mask, size_to):
    """

    :param mask:
    :param size_to:
    :return:
    """
    x = (size_to[0] - mask.shape[0])
    y = (size_to[1] - mask.shape[1])
    # return np.pad(mask, ((x/2 + x%2, x/2), (y/2  + y%2, y/2)), mode='mean')
    return np.pad(mask, ((x / 2 + x % 2, x / 2), (y / 2 + y % 2, y / 2)), mode='constant', constant_values=(0, 0))


def __gauss_derivative_kernels(size=(3, 3), sigma=None):
    m, n = [(cc - 1.) / 2. for cc in size]
    y, x = np.mgrid[-m:m + 1, -n:n + 1]
    gauss_filter = generate_gauss_2d(size, sigma)
    gx = x * gauss_filter
    gy = y * gauss_filter

    return gx, gy


def gauss_derivatives(image, filter_size=(3, 3)):
    """
    Apply derivative of a Gaussian filter on image. Client must provide image and filter size.

    :param image: to be filtered
    :param filter_size: of the filter
    :return:
    """
    gx, gy = __gauss_derivative_kernels(filter_size)

    imx = img_mp.apply_matrix_mask(image, gx)
    imy = img_mp.apply_matrix_mask(image, gy)

    return imx, imy