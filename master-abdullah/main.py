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
    freq = 1./tp

    x_out = a * np.sin(2 * np.math.pi * freq * x_in + ph)

    # plt.plot(x_in, x_out)
    # plt.show()

    g = np.zeros((f.shape[0] + abs(x_out.min()) + abs(x_out.max()), f.shape[1]))

    for y_out in xrange(f.shape[0]):
        for x in xrange(f.shape[1]):
            g[y_out + a - x_out[x], x] = f[y_out, x]

    if display:
        plot.plot_multiple_arrays([[g]], '', [''])

def v2(freq_=1, img_path="Resources/asterixGrey.jpg", display=True):
    """
    Full Sine waves in x direction
    :type img_path: path-string
    """
    f = read_image(img_path, as_array=True)
    y_in, x_in = np.ogrid[0:f.shape[0], 0:f.shape[1]]

    amp, freq, ph = 128, float(freq_)/(float(x_in.shape[1])), 0.

    y_out_diff = func_sin(x_in, amp, freq, ph).reshape(x_in.shape[1], 1).astype(int)
    y_out_main = y_in
    x_out = x_in

    g = np.zeros((y_out_main.shape[0] + abs(y_out_diff.min()) + abs(y_out_diff.max()), x_out.shape[1]))

    for i in xrange(x_out.shape[1]):
        try:
            g[y_out_main + y_out_diff.max() - y_out_diff[i], i] = f[y_out_main, i]
        except IndexError:
            print i

    if display:
        plot.plot_multiple_arrays([[g]], '', [''])

if __name__ == "__main__":
    # v1()
    # v2()
    v2(freq_=7, img_path="Resources/bauckhage.jpg")