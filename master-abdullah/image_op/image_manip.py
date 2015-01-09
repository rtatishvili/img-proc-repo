from exceptions import RuntimeError

import numpy as np

from calc import fourier_calc


def combine_magnitude_and_phase(ft_magnitude_image, ft_phase_image):
    """
    This method creates new image by combining the magnitude form **ft_magnitude_image** and phase
    from **ft_phase_image**. Client must provide two images with same resolution.
    @param ft_magnitude_image: Fourier transform of the first image
    @param ft_phase_image: Fourier transform of the second image
    @return: new image
    """
    assert ft_magnitude_image.size == ft_phase_image.size, "image_op have different resolution!"

    magnitude = fourier_calc.magnitude(ft_magnitude_image)
    phase = fourier_calc.phase(ft_phase_image)
    combined_ft_image = fourier_calc.create_complex_array(magnitude, phase)

    return np.fft.ifft2(combined_ft_image)


def insert_beyond_edge(img, img_size, size, center, mode, axis=0):
    start = center - (size / 2)
    end = center + (size / 2 + 1)

    missing_col_count_start = 0
    missing_col_count_end = 0

    if start < 0:
        missing_col_count_start = -start
        start = 0

    if end > img_size:
        missing_col_count_end = abs(img_size - end)
        end = img_size

    window = []
    if axis == 1:
        window = img[:, start:end]
    elif axis == 0:
        window = img[start:end, :]

    window = fill_missing(window, missing_col_count_start, size, 0, mode, axis)
    window = fill_missing(window, missing_col_count_end, size, 1, mode, axis)

    return window


def fill_missing(window, iteration_count, size, window_side, mode, axis):
    assert window_side == 0 or window_side == 1

    while iteration_count:
        position = window_side * (size - iteration_count)

        if mode == 'constant' or mode == 'c':
            window = np.insert(window, position, 0, axis=axis)
        elif mode == 'edge' or mode == 'e':

            edge_index = window_side * (position - 1);

            if axis == 1:
                new_edge = window[:, edge_index]
            elif axis == 0:
                new_edge = window[edge_index, :]

            window = np.insert(window, position, new_edge, axis=axis)
        else:
            raise RuntimeError

        iteration_count -= 1

    return window


def extract_window(image, size, center, mode='constant'):
    """
    This method creates new image called "window" based on the original image
    where it is small square-shape portion of image where applicable and
    some values (constant 0 by default) where it has no corresponding pixel
    in the original image (i.e. out of boundaries).
    :rtype : object
    @param image: original 2D array of an image
    @param size: size of the window as tuple (rows, cols). It should be odd values and should be equal (square)
    @param center: center coordinates of the window on th image as tuple (rows, cols)
    @param mode: how to fill the values: constant fills with zeroes, edge fills the replicas of correspoding edge
    @return: new image
    """
    # assert (size[0] % 2 != 0), "size has even value at the y dimension"
    # assert (size[1] % 2 != 0), "size has even value at the x dimension"
    # assert (size[0] == size[1]), "size should have same values on both dimensions"

    window = image
    window = insert_beyond_edge(window, window.shape[1], size[1], center[1], mode, 1)
    window = insert_beyond_edge(window, window.shape[0], size[0], center[0], mode, 0)

    return window

def extract_window_padding(image, size, mode='constant'):
    return extract_window(image, \
                          (image.shape[0] + 2 * size[0] + (image.shape[0] % 2) - 1 , \
                           image.shape[1] + 2 * size[1] + (image.shape[1] % 2) - 1), \
                          (image.shape[0] / 2, image.shape[1] / 2), mode)


def __apply_matrix_mask_on_window(image, mask, center):
    window = extract_window(image, mask.shape, center, 'edge')
    return np.sum(window * mask)


def apply_matrix_mask(image, mask):
    """
    Convolution of two dimension mask to an image. Client must provide
    2D mask and image.

    :rtype : ndarray
    :param image: to be filtered
    :param mask: to apply
    :return: new filtered image
    """

    image_with_padding = extract_window_padding(image, mask.shape, 'edge')
    result_ = np.zeros(image_with_padding.shape)
    
    for i in range(mask.shape[0]-1, image_with_padding.shape[0]):
        for j in range(mask.shape[1]-1, image_with_padding.shape[1]):
            result_[i, j] = __apply_matrix_mask_on_window(image_with_padding, mask, (i, j))
    return result_\
        [mask.shape[0] - 1:image.shape[0] + mask.shape[0] - 1, \
         mask.shape[1] - 1:image.shape[1] + mask.shape[1] - 1]


def __apply_array_mask_on_window(image, array_x, array_y, center):
    window = extract_window(image, (array_x.shape[0], array_y.shape[0]), center, 'edge')
    return window.dot(array_x).dot(array_y)


def apply_array_mask(image, array_x, array_y):
    """
    Convolution of two one dimensional mask on an image. Client must provide
    two one dimensional Gaussian masks.

    :param image: to be filtered
    :param array_x: mask in x direction
    :param array_y: mask in y direction
    :return:
    """
    image_with_padding = extract_window_padding(image, (array_x.shape[0], array_y.shape[0]), 'edge')
    
    result_ = np.zeros(image_with_padding.shape)
    
    for i in range(array_x.shape[0]-1, image_with_padding.shape[0]):
        for j in range(array_y.shape[0]-1, image_with_padding.shape[1]):
            result_[i, j] = __apply_array_mask_on_window(image_with_padding, array_x, array_y, (i, j))
                        
    return result_\
        [array_x.shape[0] - 1:image.shape[0] + array_x.shape[0] - 1, \
         array_y.shape[0] - 1:image.shape[1] + array_y.shape[0] - 1]


def apply_fourier_mask(image, mask):
    """
    Apply two dimensional Gaussian filter on image. Using the Fourier Transformations.
    Client must provide image and two dimensional mask.

    :param image: to be filtered
    :param mask:
    :return: new filtered image
    """
    mask_fft = np.fft.fftshift(np.fft.fft2(mask))
    image_fft = np.fft.fftshift(np.fft.fft2(image))
    new_image_fft = image_fft * mask_fft

    return np.fft.ifftshift(np.fft.ifft2(new_image_fft))

