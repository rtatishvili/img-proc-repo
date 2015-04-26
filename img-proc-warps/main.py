from image_op.image_io import read_image
import numpy as np
import plot
import image_op.bilinear_interpolation as bil
import image_op.bicubic_interpolation as bic # bicubic interpolation gives some artifacts in the resulting image
from math import sin
from numpy import math
from image_op import image_io


def func_sin(x_arr, amplitude, frequency, phase=0):
    return amplitude * np.sin(2 * np.math.pi * frequency * x_arr + phase)


def intermediate_wave_warp(amplitude=64, freq_=1, img_path="Resources/asterixGrey.jpg", display=True, ret=False):
    """
    Apply full sine wave in x dimension to the image
    :param amplitude: sin wave size
    :param freq_: sin wave frequency
    :param img_path: image path
    :param display: flag to display or not the result
    :param ret: flag to return or not the result
    :return: the warped image (if ret=True)
    """
    f = read_image(img_path, as_array=True)
    y_in, x_in = np.ogrid[0:f.shape[0], 0:f.shape[1]]

    # Create a sine wave, store in array of size = x dimension of the image
    amp, freq, ph = amplitude, float(freq_) / (float(x_in.shape[1])), 0.

    y_out_diff = func_sin(x_in, amp, freq, ph).reshape(x_in.shape[1], 1).astype(int)

    # this can be combined later, but left here for readability
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


def push_wave_warp(amplitude=64, wavelength=1, img=None, img_path="Resources/asterixGrey.jpg", display=True):
    """
    Full Sine Waves for y dimension
    :param amplitude: sin wave size
    :param wavelength: sin wave frequency
    :param img: image as array (if this is not provided, then img_path is needed)
    :param img_path: image path
    :param display: flag to display or not the result
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


def push_cylinder_warp(path="Resources/asterixGrey.jpg", display=True):
    """
    circumference approach
    :param path: image path
    :param display: flag to display or not the result
    :return:
    """
    im_in = read_image(path, as_array=True)

    in_radius = 0.
    out_radius = 256.

    center_in = [im_in.shape[0], 0.5 * im_in.shape[1]]
    max_radius_in = np.math.sqrt((center_in[0]) ** 2 + center_in[1] ** 2)
    min_radius_in = np.math.sqrt((center_in[0] - im_in.shape[0]) ** 2 + (center_in[1] - im_in.shape[1]) ** 2)

    x_in = np.arange(im_in.shape[1])
    y_in = np.arange(im_in.shape[0])

    phiis = x_in * (2 * np.math.pi / float(im_in.shape[1]))
    radii = in_radius + ((y_in - im_in.shape[0]) * ((out_radius - in_radius) / float(im_in.shape[0])))

    im_out = np.zeros((512, 512))
    center_out = [im_out.shape[0] / 2, im_out.shape[1] / 2]

    for i in range(radii.size):
        data = im_in[i, :]
        for j in range(phiis.size):
            d = data[j]
            x_out = center_out[1] + radii[i] * np.math.cos(phiis[j]) - 1
            y_out = center_out[0] + radii[i] * np.math.sin(phiis[j]) - 1

            im_out[int(y_out), int(x_out)] = d

    if display:
        plot.plot_multiple_arrays([[im_out, im_in]], 'anamorphosis', ['out', 'in'])
        image_io.save_array_as_gray_image(im_out, "Generated/push.jpg")
        

def pull_wave_warp(amplitude, frequency, add, img_path="Resources/bauckhage.jpg"):
    """
    Image warp with PULL approach using sin functions
    :param amplitude: a tuple of amplitude values in x and y direction respectively 
    :param frequency: a tuple of frequency values in x and y direction respectively 
    :param add: a tuple of size values in x and y direction to extend the canvas of warped image (as it may not fit in the original dimensions) 
    :param img_path: image path
    """
    original = read_image(img_path, as_array=True)
    
    # produce new wave functions with different ampliture, frequency and phase
    fx = waveFunction(amplitude[0], frequency[0] / original.shape[1], math.pi)
    fy = waveFunction(amplitude[1], frequency[1] / original.shape[0], math.pi)
    
    # prepare bigger blank canvas or warped image
    warped = np.zeros((original.shape[0] + add[0]) * (original.shape[1] + add[1])).reshape(original.shape[0] + add[0], original.shape[1] + add[1])
    
    for (y, x), value in np.ndenumerate(warped):
#         wy = y + int(fx(x))-100
#         wx = x + int(fy(y))
        wy = y + fx(x) - (add[1] / 2)
        wx = x + fy(y) - (add[0] / 2)
       
        
        if wy < 0 or wx < 0 or wy >= original.shape[0] or wx >= original.shape[1]:
            warped[y, x] = 0
        else:
            warped[y, x] = bil.resample(original, wx, wy)

    plot.plot_multiple_arrays([[original, warped]], "wave warp", ["Original", "Warped"])
    image_io.save_array_as_gray_image(warped, "Generated/bil_warp_" + str(amplitude[0]) + "_" + str(frequency[0]) + "_" + str(amplitude[1]) + "_" + str(frequency[1]) + ".jpg")


def pull_cylinder_warp(img_path="Resources/asterixGrey.jpg"):
    """
    Image warp with PULL approach using radius and angle functions
    :param img_path: image path
    """
    original = read_image(img_path, as_array=True)
    
    # produce new functions with respect to the circle center
    fr = radiusFunction(center_x=original.shape[1] / 2, center_y=original.shape[0] / 2, deadzone=0.2)
    fa = angleFunction(center_x=original.shape[1] / 2, center_y=original.shape[0] / 2)
    
    # prepare bigger blank canvas or warped image
    warped = np.zeros((original.shape[1]) * (original.shape[1])).reshape(original.shape[1], original.shape[1])
    
    for (y, x), value in np.ndenumerate(warped):
        wx = (0.5 - fa(x, y)) * original.shape[1]
        wy = (1.0 - fr(x, y)) * original.shape[0]
        
        if wy < 0 or wx < 0 or wy >= original.shape[0] or wx >= original.shape[1]:
            warped[y, x] = 0
        else:
            warped[y, x] = bil.resample(original, wx, wy)

    warped = np.fliplr(warped)
    plot.plot_multiple_arrays([[original, warped]], "cylinder warp", ["Original", "Warped"])
    image_io.save_array_as_gray_image(warped, "Generated/bil_cylinder_asterix.jpg")


def waveFunction(amplitude, frequency, phase=0.0):
    """
    :param amplitude: multiplier of sin function
    :param frequency: multiplier of angle in sin argument
    :param phase: shift of angle in sin argument
    :return: new function computing parametrized version of sin function
    """
    return lambda x: amplitude * sin(frequency * 2.0 * math.pi * x + phase)    

def radiusFunction(center_x, center_y, deadzone=0.0):
    """
    :param center_x: x coordinate of center from which the radius is computed
    :param center_y: y coordinate of center from which the radius is computed
    :param deadzone: 0 to 1.0 real value considering the portion of the inner place (small radius values) as out of image space
    :return: new function computing radius from a given center point considering the close region as deadzone
    """
    return lambda x, y: (1.0 + deadzone) * (math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2) / max(center_x, center_y)) - deadzone

def angleFunction(center_x, center_y):
    """
    :param center_x: x coordinate of center from which the angle is computed
    :param center_y: y coordinate of center from which the angle is computed
    :return: new function computing angle from a given center point
    """    
    return lambda x, y: 0.99 * math.atan2(center_y - y, center_x - x) / (2.0 * math.pi)
    
def demo():
    """
    Demo for double dim Sine Wave
    """
    print "processing PUSH wave warp..."
    push_wave_warp(wavelength=-1, amplitude=12,
         img=intermediate_wave_warp(amplitude=12, freq_=8., img_path="Resources/bauckhage.jpg", display=False, ret=True))
    print "processing PULL wave warp..."
    pull_wave_warp(amplitude=(10, 10), frequency=(8.0, 1.0), add=(64, 64))
    print "processing PUSH cylinder warp..."
    push_cylinder_warp()
    print "processing PULL cylinder warp..."
    pull_cylinder_warp()
    print "done."


if __name__ == "__main__":
    demo()
