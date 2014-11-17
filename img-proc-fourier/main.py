# imports from project
import Helper.image_io as image_io
import Helper.image_manip as image_manip
from Helper.ring_mask import RingMask
from Helper.plot import plot, plot_clean, plot_2d_gray_multi
from Helper.fourier_calc import magnitude, phase

# imports from libraries
import numpy as np

# GLOBAL VARIABLES
DEFAULT_IMAGES = ['Resources/bauckhage.jpg',
                  'Resources/clock.jpg',
                  'Resources/cat.png',
                  'Resources/asterixGrey.jpg']


def task_1_1(image_path=DEFAULT_IMAGES[0], r_min=20, r_max=40):
    image = image_io.read_image(image_path, as_array=True)
    new_image = image_manip.draw_circle(image, r_min, r_max, inside=True)
    image_io.combine_magnitude_and_phase(new_image).show()


def task_1(file_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    # Import image as a numpy.array
    input_image = image_io.read_image(file_path, as_array=True)

    # create and apply ring mask, where a ring of True lies in a sea of False
    effect_true = lambda x: x * 0  # set element to 0 where mask is True (inside the ring)
    effect_false = lambda x: x  # no change to element when mask is False (outside the ring)
    image_center = (input_image.shape[0] / 2, input_image.shape[1] / 2)  # center of ring

    ring_mask = RingMask(input_image.shape,
                         effect_true, effect_false,
                         radius_min, radius_max,
                         image_center)

    output_image_array = ring_mask.apply_mask(input_image)

    image_io.combine_magnitude_and_phase(output_image_array).show()


def task_1_2(image_path=DEFAULT_IMAGES[0], r_min=20, r_max=40):
    labels = ('Original Image', 'Fourier Transformation', 'Frequency Suppression', 'Inverse Fourier Transformation')
    im = image_io.read_image(image_path, as_array=False)
    ft = np.fft.fft2(im)
    sh = np.fft.fftshift(ft)
    image_io.save_image(np.log(np.abs(sh)), 'Generated/FourierTransformation.jpg')
    picture = image_manip.draw_circle(sh.copy(), r_min, r_max, False)
    image_io.save_image(np.log(np.abs(picture)), 'Generated/FrequencySuppression.jpg')
    inverse_p = np.fft.ifft2(picture)
    image_io.save_image(np.abs(inverse_p), 'Generated/InverseFourierTransformation.jpg')

    plot(im, sh, picture, inverse_p, labels)


def task_2(image_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    """ @author: motjuste

    :param image_path:
    :param radius_min:
    :param radius_max:
    """
    input_image_array = image_io.read_image(image_path, as_array=False)

    fft_image_array = np.fft.fft2(input_image_array)
    shift_fft_image = np.fft.fftshift(fft_image_array)

    # TODO: save the shift_fft_image

    # create and apply ring shaped freq mask, where a ring of True lies in a sea of False
    effect_true = lambda x: x  # no change to element ( = allow the freq) when mask is True (inside the ring)
    effect_false = lambda x: x * 0  # set element to zero ( = stop the freq) where mask is False (outside the ring)
    center = (shift_fft_image.shape[0] / 2, shift_fft_image.shape[1] / 2)  # center of ring

    ring_freq_mask = RingMask(shift_fft_image.shape,
                              effect_true, effect_false,
                              radius_min, radius_max,
                              center)

    out_shift_fft_image = ring_freq_mask.apply_mask(shift_fft_image)

    # TODO: save out_shift_fft_image

    # out_fft_image_array = ifftshift.numpy_ifftshift(out_shift_fft_image) # ifftshit has no effect on final image
    # out_image_array = ifft2.numpy_ifft2(out_fft_image_array)

    out_image_array = np.fft.ifft2(out_shift_fft_image)

    # plot images
    # TODO: improve plotting
    labels = ('Original Image', 'Fourier Transformation', 'Frequency Suppression', 'Inverse Fourier Transformation')
    plot(input_image_array, shift_fft_image, out_shift_fft_image, out_image_array, labels)


def task_1_3(image_path_a=DEFAULT_IMAGES[0], image_path_b=DEFAULT_IMAGES[1]):
    image_a = image_io.read_image(image_path_a, as_array=False)
    image_b = image_io.read_image(image_path_b, as_array=False)
    image_c = image_manip.combine_magnitude_and_phase(image_a, image_b)
    image_d = image_manip.combine_magnitude_and_phase(image_b, image_a)
    plot_clean(image_a, image_b, image_c, image_d,
               ('Image A', 'Image B', 'Magnitude from A, phase from B', 'Magnitude from B, phase from A'))


def task_3(image_path_list=DEFAULT_IMAGES[:2]):
    # TODO: Do you want to return the results?
    """  Task 1.3 : swap the magnitudes and phases of the fft of
                    'n'(=> 1) intensity images of same height and width
                    and display

    Implementation using numpy array operations and fft as implemented in the package
    Requires matplotlib for plotting the result

    :param image_path_list: list of strings
                           list of filepaths of images to be worked on;
                           all images should have same height and width;
                           picks from default_images when no arguments given
    """
    # import intensity images as arrays from list of file paths into a list of input images
    input_image_array_list = []
    for path in image_path_list:
        input_image_array_list.append(image_io.read_image(path, as_array=True))

    # assert that all imported image arrays are of the same shape
    assert np.all([input_image_array_list[0].shape == shape for shape in
                   [image.shape for image in input_image_array_list]]), "All images need to be of the same size"

    # calculate the fft of all images
    fft_image_list = []
    for image_array in input_image_array_list:
        fft_image_list.append(np.fft.fft2(image_array))

    # save all the calculated fft in an array
    fft_image_array = np.asarray(fft_image_list)

    mag_index, angle_index = np.ogrid[0:fft_image_array.shape[0], 0:fft_image_array.shape[0]]
    # fft_image_array.shape[0] == number of images (n)

    # column matrix of magnitudes and row matrix of angles of the fft
    magnitudes = magnitude(fft_image_array[mag_index])
    angles = phase(fft_image_array[angle_index])

    # create matrix with all the combinations of magnitudes and angles
    # matrix multiplication gives an n by n output_fft_image_array, n == number of images
    output_fft_image_array = magnitudes * (np.e ** (angles * 1J))

    # recreate images from the fft combinations and save in a list
    # each output image is an array of complex numbers
    output_image_list = []
    for output_fft_row in output_fft_image_array:
        output_image_row = []
        for each_output_fft in output_fft_row:
            output_image_row.append(np.fft.ifft2(each_output_fft))
        output_image_list.append(output_image_row)

    # extract the magnitudes of the output images
    mag_output_image_list = []
    for each_image_row in output_image_list:
        mag_output_image_row = []
        for each_image in each_image_row:
            mag_output_image_row.append(magnitude(each_image))
        mag_output_image_list.append(mag_output_image_row)

    # plot the magnitudes of the output images in an n by n matrix shape
    plot_2d_gray_multi(mag_output_image_list)


if __name__ == '__main__':
    # task_1_1()
    # task_1_2()
    # task_1_2(DEFAULT_IMAGES[1])
    # task_1_2(DEFAULT_IMAGES[3])
    # task_1_3()
    # task_1()
    # task_2()
    task_3()