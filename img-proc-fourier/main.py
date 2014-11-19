# imports from project
from image_op.image_io import read_image, save_array_as_gray_image
from image_op.image_manip import combine_magnitude_and_phase
from image_op.ring_mask import RingMask
from plot import plot_2d_gray_multi

# imports from libraries
import numpy as np
import datetime as dt

# GLOBAL VARIABLES
DEFAULT_IMAGES = ['Resources/bauckhage.jpg',
                  'Resources/clock.jpg',
                  'Resources/bauckhage.jpg',
                  'Resources/cat.png',
                  'Resources/asterixGrey.jpg']


def current_datetime_string():
    return dt.datetime.now().strftime("%Y%m%d%H%M%S")


def task_1(file_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    # Import image as a numpy.array
    input_image = read_image(file_path, as_array=True)

    # create and apply ring mask, where a ring of True lies in a sea of False

    # set element to 0 inside the ring
    func_inside_ring = lambda x: 0
    # no change to element outside the ring
    func_outside_ring = lambda x: x
    # center of ring
    image_center = (input_image.shape[0] / 2, input_image.shape[1] / 2)

    # Create a mask object containing the area of the ring and 
    # the functions to apply inside and outside the ring
    ring_mask = RingMask(input_image.shape,
                         func_inside_ring, func_outside_ring,
                         radius_min, radius_max,
                         image_center)

    # The image containing the ring (filled with black inside) 
    output_image_array = ring_mask.apply_mask(input_image)

    # save the output image
    save_array_as_gray_image(output_image_array, "Generated/task_1_" + current_datetime_string() + ".jpg")

    # plot the images for visual comparison
    plot_2d_gray_multi([[input_image, output_image_array]])

    # return the image for any further processing
    return output_image_array


def task_2(image_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    """
    :param image_path: The file path to the image on which the Fourier Transformation is applied
    :param radius_min: Minimal radius defining the inner edge of the ring
    :param radius_max: Maximal radius defining the outer edge of the ring
    """

    # Load an image as Image object
    input_image_array = read_image(image_path, as_array=True)

    # Apply fast Fourier transformation to an image
    fft_image_array = np.fft.fft2(input_image_array)

    # Shift the Fourier image to the center of the image  
    ft_image = np.fft.fftshift(fft_image_array)

    # create and apply a ring shaped frequency mask on a Fourier transform of an image
    # where the mask suppresses the frequencies outside the ring

    func_inside_ring = lambda x: x  # no change to the element, i.e. allow the frequency
    func_outside_ring = lambda x: 0  # set element to zero, i.e. suppress the frequency
    # center of ring
    center = (ft_image.shape[0] / 2, ft_image.shape[1] / 2)

    # Create a mask object containing the area of the ring and 
    # the functions to apply inside and outside the ring
    ring_freq_mask = RingMask(ft_image.shape,
                              func_inside_ring, func_outside_ring,
                              radius_min, radius_max,
                              center)

    suppressed_ft_image = ring_freq_mask.apply_mask(ft_image)

    # Apply inverse Fourier transformation on a suppressed image
    ift_image = np.fft.ifft2(suppressed_ft_image)

    # output results
    # calculate absolutes for saving and plotting
    abs_ft_image = abs(ft_image)
    abs_suppressed_ft_image = abs(suppressed_ft_image)
    abs_ift_image = abs(ift_image)

    # save output images
    current_datetime = current_datetime_string()
    save_array_as_gray_image(np.log(abs_ft_image),
                             "Generated/task_2_ft_" + current_datetime + ".jpg",
                             normalize=True)
    save_array_as_gray_image(abs_suppressed_ft_image,
                             "Generated/task_2_suppressed_ft_" + current_datetime + ".jpg",
                             normalize=True)
    save_array_as_gray_image(abs_ift_image,
                             "Generated/task_2_" + current_datetime + ".jpg")

    # plot the images for comparison
    # TODO: improve plotting
    # labels = ('Original Image', 'Fourier Transformation', 'Frequency Suppression', 'Inverse Fourier Transformation')
    # plot(input_image_array, ft_image, suppressed_ft_image, ift_image, labels)
    plot_2d_gray_multi([[input_image_array, np.log(abs_ft_image)],
                        [abs_ift_image, abs_suppressed_ft_image]])

    # return results
    return [ft_image, suppressed_ft_image, ift_image]


def task_3(image_path_list=DEFAULT_IMAGES[:2]):
    """ 
    :param image_path_list: list of paths to images to be combined
    """
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
    abs_image_list = []  # absolutes of result images; done to avoid redundant for-loops; #sorry
    for mag_index in range(len(ft_list)):
        output_image_list_row = []
        abs_image_list_row = []
        for phase_index in range(len(ft_list)):
            # Construct new image from the magnitude part of the first image and phase part of the second image
            combined_image = combine_magnitude_and_phase(ft_list[mag_index], ft_list[phase_index])

            output_image_list_row.append(combined_image)
            abs_image_list_row.append(abs(combined_image))
        output_image_list.append(output_image_list_row)
        abs_image_list.append(abs_image_list_row)

    # save output images with descriptive name
    current_datetime = current_datetime_string()
    for index_row, image_row in enumerate(abs_image_list):
        for index_col, image in enumerate(image_row):
            save_array_as_gray_image(image,
                                     "Generated/task_3_M" + str(index_row) + "_P" + str(
                                         index_col) + "_" + current_datetime + ".jpg",
                                     normalize=True)

    # plot the images for comparison
    plot_2d_gray_multi(abs_image_list)

    # return results
    return output_image_list


if __name__ == '__main__':
    task_1()
    task_2()
    # task_2(DEFAULT_IMAGES[1])
    # task_2(DEFAULT_IMAGES[3])
    # task_3(DEFAULT_IMAGES[:3])