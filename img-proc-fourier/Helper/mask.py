__author__ = 'motjuste'

import numpy as np


class Mask(object):
    # TODO: expand for multi-dimensional "images"
    def __init__(self, original_array_shape, effect_true, effect_false=lambda x: x, custom_data=None):
        """
        Initialize the Mask class
        Mask.data will be generated as per the original_array_shape, and custom_data (if provided)

        :param original_array_shape: tuple
                                     with the array.shape of the array that the mask is to be applied to
                                     Mask.data will be of this shape
        :param effect_true: function (mandatory)
                            the effect of the Mask on the elements of the array where Mask.data is True
                            currently supported only functions with one argument,
                                and one return of the same type as argument
                            call without brackets, mapping going on here
        :param effect_false: function (not mandatory; default=lambda x: x)
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
        self.effect_true = effect_true
        self.effect_false = effect_false

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
        Apply the Mask.effect_true and Mask.effect_false on a copy of the array provided, and return the copy
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
        # apply effect_true where Mask.data is True
        array_copy[self.data] = map(self.effect_true, array_copy[self.data])

        # apply effect_false where Mask.data is False
        array_copy[np.invert(self.data)] = map(self.effect_false, array_copy[np.invert(self.data)])

        return array_copy


# =============================Tests=================================================
def func(size):
    condition_list = []
    for i in range(size[0]):
        condition_sublist = []
        for j in range(size[1]):
            condition_sublist.append(i == j)
        condition_list.append(condition_sublist)

    return np.array(condition_list)


def test_set_data_from_2d_condition():
    # --------- Test with euclidean
    mask = Mask(np.ones((5, 5)), lambda x: x * 1, lambda x: x * 0)

    from distance import euclidean_2d_array

    condition = euclidean_2d_array(mask.shape, (mask.shape[0] / 2, mask.shape[1] / 2)) > 1

    mask.set_data(condition)
    # print mask.data

    # --------- Test with sample func
    mask2 = Mask(np.ones((5, 5)), lambda x: x * 1, lambda x: x * 0)
    condition2 = func(mask2.shape)
    mask2.set_data(condition2)
    # print mask2.data


def test_apply_mask():
    a = np.random.randint(1, 30, (5, 5))
    # mask = Mask(a, lambda x: x*1, lambda x: x*0)
    # print mask.data

    condition = func(a.shape)
    # mask.set_data_from_2d_condition(condition)
    mask = Mask(a, lambda x: x * 1, lambda x: x * 0, condition)
    # print mask.data

    a2 = mask.apply_mask(a)
    # print a2

    # ========Test with Image=========================
    from image_io import read_image as import_image
    from distance import euclidean_2d_array
    import sys

    input_str = ""

    while input_str == "" and input_str != "q" and input_str != "d":
        input_str = raw_input("Enter full path to image : ")
        input_str = input_str.strip()

    if input_str == "q":
        sys.exit()
    elif input_str == "d":  # default path
        input_str = "/home/abdullah/Studies/MI/imageproc/Projects/imgproc-project-1/Resources/Images/bauckhage.jpg"

    fp = input_str.split()[0]

    im_2d_array = import_image(fp)

    condition1 = euclidean_2d_array(im_2d_array.shape, [im_2d_array.shape[0] / 2, im_2d_array.shape[1] / 2]) > 25
    mask1 = Mask(im_2d_array, lambda x: True, lambda x: False, condition1)

    condition2 = euclidean_2d_array(im_2d_array.shape, [im_2d_array.shape[0] / 2, im_2d_array.shape[1] / 2]) > 55
    mask2 = Mask(im_2d_array, lambda x: True, lambda x: True, condition2)

    mask_ring = Mask(im_2d_array, lambda x: x * 0, lambda x: x, mask1.data != mask2.data)
    im_array_out = mask_ring.apply_mask(im_2d_array)

    from visual_test import plot_2d_gray

    plot_2d_gray([im_2d_array, im_array_out])


if __name__ == "__main__":
    test_set_data_from_2d_condition()
    test_apply_mask()