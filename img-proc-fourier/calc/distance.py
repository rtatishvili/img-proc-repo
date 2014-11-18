# imports from libraries
from numpy import sqrt, ogrid


def euclidean_2d_array(array_shape, from_el_loc):
    """
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

