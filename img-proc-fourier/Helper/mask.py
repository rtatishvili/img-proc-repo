__author__ = 'motjuste'

import numpy as np


class Mask(object):
    # TODO: expand for multi-dimensional "images"
    def __init__(self, original_array_shape, func_true, func_false=lambda x: x, custom_data=None):
        """
        Initialize the Mask class
        Mask.data will be generated as per the original_array_shape, and custom_data (if provided)

        :param original_array_shape: tuple
                                     with the array.shape of the array that the mask is to be applied to
                                     Mask.data will be of this shape
        :param func_true: function (mandatory)
                            the effect of the Mask on the elements of the array where Mask.data is True
                            currently supported only functions with one argument,
                                and one return of the same type as argument
                            call without brackets, mapping going on here
        :param func_false: function (not mandatory; default=lambda x: x)
                             the effect of the Mask on the elements of the array where Mask.data is False
                             currently supported only functions with one argument,
                                and one return of the same type as argument
                             no effect by default
                             call without brackets, mapping going on here
        :param custom_data: boolean array of the same shape as original_array_shape (not mandatory; default=None)
                            if None, Mask.data is set as a boolean matrix with all values False
                            else, custom_data will be set as Mask.data
        """
        self.shape = original_array_shape
        self.func_true = func_true
        self.func_false = func_false

        # initialize Mask.data as an all False boolean numpy.array
        self.data = self.init_data()

        # set Mask.data to custom_data, if provided
        if not custom_data is None:
            self.set_data(custom_data)

    def init_data(self):
        """
        Returns the data that the Mask.data can be initialized with

        :return: an all False boolean numpy.array of size Mask.shape
        """
        return np.zeros(self.shape, dtype=bool)

    def set_data(self, data):
        """
        User callable method to set "custom" Mask.data
        Yes, the burden is on the user to set the data, but, be creative!

        :param data: 2D numpy.array of the same size as Mask.data, and of data type bool;
                     The Boolean matrix that will define the mask
        """
        assert data.shape == self.shape, "data should be an array of the same shape as Mask"
        assert data.dtype == bool, "data should be an array of data type bool"

        self.data = data

    def apply_mask(self, array):
        """
        Apply the Mask.func_true and Mask.func_false on a copy of the array provided, and return the copy
        Currently Mask.effect_* are basically single point operations (functions with one argument),
        the effects are applied to each element of the input array, depending on Mask.data
        One will have to be creative for more complex effects

        :param array: numpy.array
                      array must be of the same size as the Mask
        :return: numpy.array
                 result of the Mask.effect_* applied to the input array
        """
        # assert that the array and Mask.data are of the same size
        assert array.shape == self.shape, "array and mask should be of the same shape"

        array_copy = array.copy()

        # Applying mask
        # apply func_true where Mask.data is True
        array_copy[self.data] = map(self.func_true, array_copy[self.data])

        # apply func_false where Mask.data is False
        array_copy[np.invert(self.data)] = map(self.func_false, array_copy[np.invert(self.data)])

        return array_copy

