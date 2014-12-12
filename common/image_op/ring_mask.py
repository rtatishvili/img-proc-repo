from image_op.mask import Mask
from calc.distance import euclidean_2d_array


class RingMask(Mask):
    def __init__(self, original_array_shape, func_true, func_false, radius_inner, radius_outer, center):
        """
        Initialize a RingMask object of class Mask
        where the array RingMask.data has True forming
        a ring with radius_inner and radius_outer around center
        in a pool of False

        Refer Mask class for details

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
        :param radius_inner: float
                             inner radius of the ring shape
        :param radius_outer: float
                             outer radius of the ring shape
        :param center: tuple
                       location of the center of the ring
        """
        super(RingMask, self).__init__(original_array_shape,
                                       func_true, func_false=func_false)
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.center = center
        self.set_data(self.gen_ring_data())

    def gen_ring_data(self):
        """
        Returns an array that has True forming
        a ring with RingMask.radius_inner and RingMask.radius_outer around RingMask.center
        in a pool of False

        :return: boolean array of shape as original_array_shape
        """
        # create a dist_array of the shape of RingMask.shape
        # where each element value is the Euclidean distance of the element
        # from the center of RingMask.shape
        dist_array = euclidean_2d_array(self.shape, self.center)

        # *_circle_data has True elements forming a circle, surrounded by False elements
        inner_circle_data = dist_array >= self.radius_inner
        outer_circle_data = dist_array >= self.radius_outer

        # since *_circle_data is boolean
        ring_data = outer_circle_data != inner_circle_data

        return ring_data

