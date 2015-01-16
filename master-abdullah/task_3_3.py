__author__ = 'myrmidon'

from image_op.image_io import read_image
import numpy as np
from plot import plot_multiple_arrays
import matplotlib.pyplot as plt
import matplotlib.animation as nim8
import time


def v0(path='Resources/bauckhage.jpg', display=True):
    im_in = read_image(path, as_array=True)
    # im_in = np.arange(25).reshape(5, 5)

    # NOTES:
    # The resulting image should be a square ??????
    # The lower edge of the original image is mapped to the inner circle of the resulting image
    # The upper edge of the original image is mapped to the outer circle of the resulting image
    # Strictly speaking, both the edges will be mapped to the lower/upper circumference of the
    #   hypothetical cylinder
    # Strictly speaking, the lower edge of the original image
    #   will be mapped to the 'actual' circumference of the hypothetical cylinder
    # Strictly speaking, the ratio of the inner and outer circumferences, hence the radii, depend on the
    #   angle made between the height of the original image and the height of the cylinder.
    #   I think the angle can be calculated as atan(height_cylinder / height_original_image)
    # Strictly speaking, then wouldn't reconstructing the first demo effect cause problems,
    #   where the cylinder seems to be of zero radius?
    #       But of finite Height!!!!
    # Strictly speaking, WTF ABOUT WE ARE ONLY GOING TO LOOK AT THE REFLECTION!!!?????
    #   I am probably going to over-complicate this shit.

    angles = range(im_in.shape[1])
    radii = range(im_in.shape[0])

    im_out_dict = {}

    for angle in angles:
        temp_radial_line = {}
        for radius in radii:
            temp_radial_line[radius] = im_in[angle, radius]
        im_out_dict.update({angle: temp_radial_line})

    im_out = np.zeros((256 * 2 + 1, 256 * 2 + 1))

    center_out = (im_out.shape[0]/2, im_out.shape[1]/2)

    for angle in angles:
        for radius in radii:
            out_coord = [center_out[0] + radius * np.math.sin(angle * 2 * np.math.pi/float(len(angles))),
                         center_out[1] + radius * np.math.cos(angle * 2 * np.math.pi/float(len(angles)))]
            im_out[int(out_coord[0]), int(out_coord[1])] = im_out_dict[angle][radius]

    check_out = im_out[256, 256:]
    check_in = im_in[:, 0]

    if display:
        plot_multiple_arrays([[im_in, im_out]], 'Task 3.3', ['Input', 'Output'])

def v0_1(path="Resources/bauckhage.jpg", display=True):
    im_in = read_image(path, as_array=True)
    # im_in = np.array([[x + y for x in range(128)] for y in range(128)]).reshape(128, 128)
    # im_in = np.array(range(256) * 256).reshape(256, 256)

    angles = range(im_in.shape[1])
    radii = range(im_in.shape[0])

    out_outer_circ = im_in.shape[0]
    out_radius = int(out_outer_circ / 2 * np.math.pi)

    im_out = np.ones((2*out_radius + 1, 2*out_radius + 1)) * 128
    # im_out = np.zeros((256 * 2 + 1, 256 * 2 + 1))

    center_out = (im_out.shape[0]/2, im_out.shape[1]/2)
    white = True
    for angle in angles:
        radial_line = np.ogrid[0:len(radii)]
        # values_in = im_in[:, angle]
        x_radial_line = center_out[1] + radial_line * np.math.sin((angle) * 2 * np.math.pi / float(len(angles)))
        y_radial_line = center_out[0] + radial_line * np.math.cos((angle) * 2 * np.math.pi / float(len(angles)))

        # im_out[y_radial_line.astype(int).reshape(y_radial_line.shape[0], 1), x_radial_line.astype(int)] = im_in[:, angle]
        # im_out[y_radial_line.astype(int).reshape(y_radial_line.shape[0], 1), x_radial_line.astype(int)] = white * 256
        # white = not white

        for y in y_radial_line:
            for x in x_radial_line:
                im_out[y, x] = white * 256

    print im_out[center_out[0], center_out[1]:]

    if display:
        plot_multiple_arrays([[im_in, im_out]], 'cylindrical anamorphosis', ['input', 'output'])

def v0_2(path="Resources/bauckhage.jpg", display=True):
    im_in = read_image(path, as_array=True)

    im_out = np.ones((512 * 2 + 1, 512 * 2 + 1)) * 100
    radius = 256
    center = (im_out.shape[0]/2, im_out.shape[1]/2)
    data = np.array([x%2 for x in xrange(256)])
    x = np.arange(radius)
    y = np.arange(radius).reshape(radius, 1)
    # print x, y

    for angle in xrange(3):
        y_out = center[0] + y * np.math.sin(angle * np.math.pi/180.)
        x_out = center[1] + x * np.math.cos(angle * np.math.pi/180.)

        # im_out[y_out.astype(int), x_out.astype(int)] = 256

        for i in xrange(x_out.size):
            for j in xrange(y_out.size):
                im_out[int(y_out[j]), int(x_out[i])] = 256

    if display:
        plot_multiple_arrays([[im_in, im_out]], 'cylindrical anamorphosis', ['input', 'output'])


def v0_3(path="Resources/bauckhage.jpg", display=True):
    """
    circumference approach
    :param path:
    :param display:
    :return:
    """
    im_in = read_image(path, as_array=True)

    in_radius = 0.
    out_radius = 256.

    # center_in = [np.math.sqrt(in_radius ** 2 - (0.5 * im_in.shape[1]) ** 2), 0.5 * im_in.shape[1]]
    center_in = [im_in.shape[0], 0.5 * im_in.shape[1]]
    max_radius_in = np.math.sqrt((center_in[0]) ** 2 + center_in[1] ** 2)
    min_radius_in = np.math.sqrt((center_in[0] - im_in.shape[0]) ** 2 + (center_in[1] - im_in.shape[1]) ** 2)

    x_in = np.arange(im_in.shape[1])
    y_in = np.arange(im_in.shape[0])

    phiis = x_in * (2 * np.math.pi / float(im_in.shape[1]))
    radii = in_radius + ((y_in - im_in.shape[0]) * ((out_radius - in_radius) / float(im_in.shape[0])))


    # im_out = np.zeros((r_out.max() * 2, r_out.max() * 2))
    im_out = np.zeros((512, 512))
    center_out = [im_out.shape[0]/2, im_out.shape[1]/2]

    for i in range(radii.size):
        data = im_in[i, :]
        for j in range(phiis.size):
            d = data[j]
            x_out = center_out[1] + radii[i] * np.math.cos(phiis[j]) - 1
            y_out = center_out[0] + radii[i] * np.math.sin(phiis[j]) - 1

            im_out[int(y_out), int(x_out)] = d

    # x_out = center_out[1] + radii * np.cos(phiis)
    # y_out = center_out[0] + radii * np.sin(phiis)

    # from scipy.ndimage.interpolation import affine_transform
    # matr = np.array([[0, out_radius / float(im_in.shape[1])],
    #                  [2 * np.math.pi / float(im_in.shape[1]), 0]])
    #
    # im_out_aff = affine_transform(im_in, matr)


    if display:
        plot_multiple_arrays([[im_out, im_in]], 'anamorphosis', ['out', 'in'])

if __name__ == "__main__":
    # v0()
    # v0_1()
    # v0_2()
    v0_3(path="Resources/asterixGrey.jpg")