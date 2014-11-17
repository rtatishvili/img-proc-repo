# imports from Helper package
from Assert import Assert

# imports from libraries
from numpy import sqrt, ogrid


def euclidean_2d_array(array_shape, from_el_loc):
    """ @author: motjuste
    Returns an array of shape = array_shape,
    where each element is a float of value equal to the
    Euclidean distance of the element
    from the from_el_loc

    :param array_shape: tuple
                        shape of the array
    :param from_el_loc: tuple
                        location of the element from where the Euclidean distance is to be calculated
    :return: array of shape as array_shape
    """
    # assertions for inputs
    assert len(array_shape) == len(from_el_loc), "array shape and element location dimension mismatch"
    assert len(array_shape) == 2, "Only 2D supported in the function"  # TODO: _ndChanges

    # create col and row arrays with values of all possible x and y coordinates respectively
    col_array, row_array = ogrid[0:array_shape[0], 0:array_shape[1]]

    # generate the return array by matrix multiplication
    return sqrt((col_array - from_el_loc[0]) ** 2 + (row_array - from_el_loc[1]) ** 2)


def euclidean_2points(x, y):
    """ @author: kostadin
    This method calculates Euclidean distance between two points.
    The client must provide array of the coordinate of the points.
    @param x: first point
    @param y: second point
    @return: distance between x and y
    """
    Assert.isTrue(len(x) == len(y), "Arrays have different length")
    a = 0.
    for i in range(len(x)):
        a += (x[i] - y[i]) ** 2

    return sqrt(a)


# =============================Tests=================================================
def visual_test_euclidean():
    """


    """
    import numpy as np

    im = np.random.randint(0, 255, (750, 256))

    im_center = (im.shape[0] / 2, im.shape[1] / 2)
    dist_array = euclidean_2d_array(im.shape, im_center)

    # Mask
    mask_inner = dist_array < 25
    mask_outer = dist_array < 55
    mask = mask_inner != mask_outer

    # Apply Mask
    im1 = im.copy()
    im1[mask] = 0

    # Display as plots
    from visual_test import plot_2d_gray

    plot_2d_gray([im, im1])


if __name__ == "__main__":
    visual_test_euclidean()