from image_op.image_io import read_image
import numpy as np
import plot
import matplotlib.pyplot as plt


def func_sin(x_arr, amplitude, frequency, phase=0):
    return amplitude * np.sin(2 * np.math.pi * frequency * x_arr + phase)


def v1(display=True):
    f = read_image("Resources/bauckhage.jpg", as_array=True)

    x_in = np.arange(f.shape[1])

    a = 15
    ph = 0.
    tp = f.shape[0] * 2.
    freq = 1. / tp

    x_out = a * np.sin(2 * np.math.pi * freq * x_in + ph)

    # plt.plot(x_in, x_out)
    # plt.show()

    g = np.zeros((f.shape[0] + abs(x_out.min()) + abs(x_out.max()), f.shape[1]))

    for y_out in xrange(f.shape[0]):
        for x in xrange(f.shape[1]):
            g[y_out + a - x_out[x], x] = f[y_out, x]

    if display:
        plot.plot_multiple_arrays([[g]], '', [''])


def v2(amplitude=64, freq_=1, img_path="Resources/asterixGrey.jpg", display=True, ret=False):
    """
    Full sine wave in x dimension
    :param amplitude:
    :param freq_:
    :param img_path:
    :param display:
    :param ret:
    :return:
    """
    f = read_image(img_path, as_array=True)
    y_in, x_in = np.ogrid[0:f.shape[0], 0:f.shape[1]]

    # Create a sine wave, store in array of size = x dimension of the image
    amp, freq, ph = amplitude, float(freq_) / (float(x_in.shape[1])), 0.

    y_out_diff = func_sin(x_in, amp, freq, ph).reshape(x_in.shape[1], 1).astype(int)

    # this stuff can be combined later, but here for readability
    y_out_main = y_in
    x_out = x_in

    # initialize the resulting image with all zeros
    # incorporate the possible change in size of the image in y dim due to the sine wave
    g = np.zeros((y_out_main.shape[0] + abs(y_out_diff.min()) + abs(y_out_diff.max()), x_out.shape[1]))

    for i in xrange(x_out.shape[1]):
        try:
            # for a particular i in x, set values for all y in g from f
            g[y_out_main + y_out_diff.max() - y_out_diff[i], i] = f[y_out_main, i]
        except IndexError:
            print i

    if display:
        plot.plot_multiple_arrays([[g]], '', [''])

    if ret:
        return g


def v2_1(amplitude=64, wavelength=1, img=None, img_path="Resources/asterixGrey.jpg", display=True):
    """
    Full Sine Waves for y dimension
    :param amplitude:
    :param wavelength:
    :param img:
    :param img_path:
    :param display:
    :return:
    """
    if img is None:
        f = read_image(img_path, as_array=True)
    else:
        f = img
    y_in, x_in = np.ogrid[0:f.shape[0], 0:f.shape[1]]

    # Create a sine wave, store in array of size = y dimension of the image
    amp, freq, ph = amplitude, 1. / float(y_in.shape[0] * wavelength), 0.

    x_out_diff = func_sin(y_in.reshape(y_in.shape[1], y_in.shape[0]), amp, freq, ph).reshape(1, y_in.shape[0]).astype(
        int)

    # just for readability
    x_out_main = x_in
    y_out = y_in

    # initialize output image, keeping in mind the size of the image in x dim may change
    g = np.zeros((y_out.shape[0], x_out_main.shape[1] + abs(x_out_diff.max()) + abs(x_out_diff.min())))

    for j in xrange(y_out.shape[0]):
        try:
            # for each row in y dim of g and f, copy all values from f and place appropriately in g
            g[j, x_out_main + x_out_diff.max() - x_out_diff[0, j]] = f[j, x_out_main]
        except ValueError:
            print j

    if display:
        plot.plot_multiple_arrays([[g]], '', [''])


def demo(display=False):
    """
    Demo for double dim Sine Wave
    :param display:
    :return:
    """
    v2_1(wavelength=-1, amplitude=12, display=display,
         img=v2(amplitude=12, freq_=8., img_path="Resources/bauckhage.jpg", display=False, ret=True))


if __name__ == "__main__":
    # v1()
    # v2()
    # v2(freq_=7, img_path="Resources/bauckhage.jpg")
    # v2_1(wavelength=1/2.)
    demo(display=True)