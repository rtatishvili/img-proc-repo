# imports from project
from common.image_op.image_io import read_image, save_array_as_gray_image
from common.calc.fourier_calc import magnitude
from common.image_op.image_manip import combine_magnitude_and_phase
from common.image_op.ring_mask import RingMask
from common.plot import plot_multiple_arrays

# imports from libraries
import numpy as np
import sys

# GLOBAL VARIABLES
DEFAULT_IMAGES = ['Resources/bauckhage.jpg',
                  'Resources/clock.jpg',
                  'Resources/bauckhage.jpg',
                  'Resources/cat.png',
                  'Resources/asterixGrey.jpg']


def task_1(image_path, radius_min, radius_max):
    # Import image as a numpy.array
    """
    :param image_path: file path to the image on which the ring is to be drawn
    :param radius_min: minimal radius of the ring
    :param radius_max: maximal radius of the ring
    :return: array of image data with the applied circle
    """
    print "####################################"
    print
    print "Running Task 1"
    input_image = read_image(image_path, as_array=True)

    # create and apply ring mask, where a ring of True lies in a sea of False
    func_inside_ring = lambda x: 0  # set element to 0 inside the ring
    func_outside_ring = lambda x: x  # no change to element outside the ring
    image_center = (input_image.shape[0] / 2, input_image.shape[1] / 2)  # used as center of ring

    # Create a mask object containing the area of the ring and 
    # the functions to apply inside and outside the ring
    ring_mask = RingMask(input_image.shape,
                         func_inside_ring, func_outside_ring,
                         radius_min, radius_max,
                         image_center)

    # The image containing the ring (filled with black inside) 
    output_image_array = ring_mask.apply_mask(input_image)

    # Save the output image
    save_array_as_gray_image(output_image_array, "Generated/task_1.jpg")

    print "Output Image saved at :",
    print "Generated/task_1.jpg"

    # plot the images for visual comparison
    plot_multiple_arrays([[input_image, output_image_array]],
                         "Task 1", ["Input Image: g", "Output Image: g`"])

    # return the image for any further processing
    return output_image_array


def task_2(image_path, radius_min, radius_max):
    """
    :param image_path: The file path to the image on which the Fourier Transformation is applied
    :param radius_min: Minimal radius defining the inner edge of the ring
    :param radius_max: Maximal radius defining the outer edge of the ring
    :return: list of image arrays
            [FT of input image, suppressed FT, image from the suppressed FT]
    """
    print "####################################"
    print
    print "Running Task 2"
    input_image_array = read_image(image_path, as_array=True)

    # Apply fast Fourier transformation to an image
    ft_image_array = np.fft.fft2(input_image_array)

    # Shift the Fourier image to the center of the image  
    shift_ft_image = np.fft.fftshift(ft_image_array)

    # create and apply a ring shaped frequency mask on a Fourier transform of an image
    # where the mask suppresses the frequencies outside the ring
    func_inside_ring = lambda x: x  # no change to the element, i.e. allow the frequency
    func_outside_ring = lambda x: 0  # set element to zero, i.e. suppress the frequency
    center = (shift_ft_image.shape[0] / 2, shift_ft_image.shape[1] / 2)  # center of the ring

    # Create and apply the RingMask
    ring_freq_mask = RingMask(shift_ft_image.shape,
                              func_inside_ring, func_outside_ring,
                              radius_min, radius_max,
                              center)
    suppressed_ft_image = ring_freq_mask.apply_mask(shift_ft_image)

    # inverse the shifting of fft
    shift_suppressed_ft_image = np.fft.ifftshift(suppressed_ft_image)
    # image from the suppressed FT
    output_image = np.fft.ifft2(shift_suppressed_ft_image)

    # output results
    # calculate absolutes for saving and plotting
    abs_ft_image = magnitude(shift_ft_image)
    abs_suppressed_ft_image = magnitude(suppressed_ft_image)
    abs_output_image = magnitude(output_image)

    # save output images
    save_array_as_gray_image(np.log(abs_ft_image),
                             "Generated/task_2_ft.jpg",
                             normalize=True)
    save_array_as_gray_image(abs_suppressed_ft_image,
                             "Generated/task_2_suppressed_ft.jpg",
                             normalize=True)
    save_array_as_gray_image(abs_output_image,
                             "Generated/task_2.jpg")

    print "Output Images saved at :"
    print "Generated/task_2_ft.jpg"
    print "Generated/task_2_suppressed_ft.jpg"
    print "Generated/task_2.jpg"

    # plot the images for comparison
    labels = ('Original Image: g', 'FT of g: G', 'IFT of G`: g`', 'Suppressed FT G: G`')
    plot_multiple_arrays([[input_image_array, np.log(abs_ft_image)],
                          [abs_output_image, abs_suppressed_ft_image]],
                         "Task 2", labels)

    # return results
    return [shift_ft_image, suppressed_ft_image, output_image]


def task_3(image_path_list):
    """
    :param image_path_list: list of paths to images to be combined
    :return: 2D list of combined image arrays, accessed as list[x][y], where, while combining:
            x: image array in image_path_list whose magnitude was taken
            y: image array in image_path_list whose phase was taken
    """
    print "####################################"
    print
    print "Running Task 3"
    # import intensity images as Images from list of file paths into a list of input images
    input_image_array_list = []
    for path in image_path_list:
        input_image_array_list.append(read_image(path, as_array=True))

    # assert that all imported image arrays are of the same shape
    assert np.all([input_image_array_list[0].shape == shape for shape in
                   [image.shape for image in input_image_array_list]]), "All images need to be of the same size"

    # calculate the fft of all images
    ft_list = []
    for image_array in input_image_array_list:
        ft_list.append(np.fft.fft2(image_array))

    # combine the phase and magnitudes of images
    output_image_list = []
    for mag_index in range(len(ft_list)):
        output_image_list_row = []
        for phase_index in range(len(ft_list)):
            # Construct new image from the magnitude part of the first image and phase part of the second image
            combined_image = combine_magnitude_and_phase(ft_list[mag_index], ft_list[phase_index])
            output_image_list_row.append(combined_image)
        output_image_list.append(output_image_list_row)

    # save output images with descriptive name
    abs_image_list = []  # absolutes of result images; done to avoid redundant for-loops; #sorry
    labels = []  # for plotting
    print "Output Images saved at :"

    for index_row, image_row in enumerate(output_image_list):
        abs_image_list_row = []
        for index_col, image in enumerate(image_row):
            abs_image = magnitude(image)
            save_array_as_gray_image(abs_image,
                                     "Generated/task_3_M" + str(index_row) + "_P" + str(
                                         index_col) + ".jpg",
                                     normalize=True)
            print "Generated/task_3_M" + str(index_row) + "_P" + str(index_col) + ".jpg"
            abs_image_list_row.append(abs_image)
            labels.append("Mag(" + str(index_row) + "), Phase(" + str(index_col) + ")")  # for plotting
        abs_image_list.append(abs_image_list_row)

    # plot the images for comparison
    plot_multiple_arrays(abs_image_list, "Task 3", labels)

    # return results
    return output_image_list


# ============== Interactive Usage =============================
def main(argv):
    task = ''
    if len(argv) == 1:
        task += 'all'
    else:
        task += argv[1]

    if task == '1':
        task_1(image_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.)
    elif task == '2':
        task_2(image_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.)
    elif task == '3':
        task_3(image_path_list=DEFAULT_IMAGES[:2])
    elif task == '3x3':
        task_3(image_path_list=DEFAULT_IMAGES[:3])
    elif task == 'all':
        task_1(image_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.)
        task_2(image_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.)
        task_3(image_path_list=DEFAULT_IMAGES[:2])
    else:
        print "Unrecognized script parameter"
        sys.exit()


if __name__ == "__main__":
    main(sys.argv)