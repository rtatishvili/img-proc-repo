# imports from project
from Helper.image_io import read_image, save_image, create_image
from Helper.image_manip import draw_circle, combine_magnitude_and_phase
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

# [Revaz] I think The version with ring_mask is better because
# it allows to manipulate the inside and outside regions with lambda functions

# def task_1_1(image_path=DEFAULT_IMAGES[0], r_min=20, r_max=40):
#     
#     image = read_image(image_path, as_array=True)
#     new_image = draw_circle(image, r_min, r_max, inside=True)
#     create_image(new_image).show()


def task_1(file_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    # Import image as a numpy.array
    input_image = read_image(file_path, as_array=True)

    # create and apply ring mask, where a ring of True lies in a sea of False
    
    # set element to 0 inside the ring
    func_inside_ring = lambda x: x * 0
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

    # create an image object from array to show on a screen
    create_image(output_image_array).show()


# [Revaz] The useful parts are copied to the second version
# where, again, ring mask is used instead of draw circle

# def task_1_2(image_path=DEFAULT_IMAGES[0], r_min=20, r_max=40):
#     labels = ('Original Image', 'Fourier Transformation', 'Frequency Suppression', 'Inverse Fourier Transformation')
#     im = read_image(image_path, as_array=False)
#     ft = np.fft.fft2(im)
#     sh = np.fft.fftshift(ft)
#     save_image(np.log(np.abs(sh)), 'Generated/FourierTransformation.jpg')
#     picture = draw_circle(sh.copy(), r_min, r_max, False)
#     save_image(np.log(np.abs(picture)), 'Generated/FrequencySuppression.jpg')
#     inverse_p = np.fft.ifft2(picture)
#     save_image(np.abs(inverse_p), 'Generated/InverseFourierTransformation.jpg')
# 
#     plot(im, sh, picture, inverse_p, labels)


def task_2(image_path=DEFAULT_IMAGES[0], radius_min=25., radius_max=55.):
    """ @author: motjuste

    :param image_path: The file path to the image on which the Fourier Transformation is applied
    :param radius_min: Minimal radius defining the inner edge of the ring
    :param radius_max: Maximal radius defining the outer edge of the ring
    """
    
    # Load an image as Image object
    input_image_array = read_image(image_path, as_array=False)

    # Apply fast Fourier transformation to an image
    fft_image_array = np.fft.fft2(input_image_array)
    
    # Shift the Fourier image to the center of the image  
    ft_image = np.fft.fftshift(fft_image_array)

    # save the shifted Fourier transform image
    save_image(np.log(np.abs(ft_image)), 'Generated/FourierTransformation.jpg')

    # create and apply a ring shaped frequency mask on a Fourier transform of an image
    # where the mask suppresses the frequencies outside the ring
    
    # no change to the element, i.e. allow the frequency
    func_inside_ring = lambda x: x  
    # set element to zero, i.e. suppress the frequency
    func_outside_ring = lambda x: x * 0
    # center of ring
    center = (ft_image.shape[0] / 2, ft_image.shape[1] / 2)  

    # Create a mask object containing the area of the ring and 
    # the functions to apply inside and outside the ring
    ring_freq_mask = RingMask(ft_image.shape,
                              func_inside_ring, func_outside_ring,
                              radius_min, radius_max,
                              center)

    suppressed_ft_image = ring_freq_mask.apply_mask(ft_image)

    # save the suppressed version shifted Fourier transform image
    save_image(np.log(np.abs(suppressed_ft_image)), 'Generated/FrequencySuppression.jpg')


    # out_fft_image_array = ifftshift.numpy_ifftshift(suppressed_ft_image) # ifftshit has no effect on final image
    # ift_image = ifft2.numpy_ifft2(out_fft_image_array)

    # Apply inverse Fourier transformation on a suppressed image
    ift_image = np.fft.ifft2(suppressed_ft_image)

    # plot images
    # TODO: improve plotting
    labels = ('Original Image', 'Fourier Transformation', 'Frequency Suppression', 'Inverse Fourier Transformation')
    plot(input_image_array, ft_image, suppressed_ft_image, ift_image, labels)

# [Revaz] For the third task I preferred the clear solution which is easy to explain and document.
# Generalized version is saved into another branch of the project for the future use, when needed. 

def task_3(image_path_a=DEFAULT_IMAGES[0], image_path_b=DEFAULT_IMAGES[1]):
    """ @author: kostadin

    :param image_path_a: The file path to the first image
    :param image_path_b: The file path to the second image
    """
    
    # Load first image as Image object
    image_a = read_image(image_path_a, as_array=False)
    # Load second image as Image object
    image_b = read_image(image_path_b, as_array=False)
    # Construct new image from the magnitude part of the first image and phase part of the second image
    image_c = combine_magnitude_and_phase(image_a, image_b)
    # Construct new image from the magnitude part of the second image and phase part of the first image
    image_d = combine_magnitude_and_phase(image_b, image_a)
    
    # display the images for comparison
    plot_clean(image_a, image_b, image_c, image_d,
               ('Image A', 'Image B', 'Magnitude from A, phase from B', 'Magnitude from B, phase from A'))


# def task_3(image_path_list=DEFAULT_IMAGES[:2]):
#     # TODO: Do you want to return the results?
#     """  Task 1.3 : swap the magnitudes and phases of the fft of
#                     'n'(=> 1) intensity images of same height and width
#                     and display
# 
#     Implementation using numpy array operations and fft as implemented in the package
#     Requires matplotlib for plotting the result
# 
#     :param image_path_list: list of strings
#                            list of filepaths of images to be worked on;
#                            all images should have same height and width;
#                            picks from default_images when no arguments given
#     """
#     # import intensity images as arrays from list of file paths into a list of input images
#     input_image_array_list = []
#     for path in image_path_list:
#         input_image_array_list.append(read_image(path, as_array=True))
# 
#     # assert that all imported image arrays are of the same shape
#     assert np.all([input_image_array_list[0].shape == shape for shape in
#                    [image.shape for image in input_image_array_list]]), "All images need to be of the same size"
# 
#     # calculate the fft of all images
#     fft_image_list = []
#     for image_array in input_image_array_list:
#         fft_image_list.append(np.fft.fft2(image_array))
# 
#     # save all the calculated fft in an array
#     fft_image_array = np.asarray(fft_image_list)
# 
#     mag_index, angle_index = np.ogrid[0:fft_image_array.shape[0], 0:fft_image_array.shape[0]]
#     # fft_image_array.shape[0] == number of images (n)
# 
#     # column matrix of magnitudes and row matrix of angles of the fft
#     magnitudes = magnitude(fft_image_array[mag_index])
#     angles = phase(fft_image_array[angle_index])
# 
#     # create matrix with all the combinations of magnitudes and angles
#     # matrix multiplication gives an n by n output_fft_image_array, n == number of images
#     output_fft_image_array = magnitudes * (np.e ** (angles * 1J))
# 
#     # recreate images from the fft combinations and save in a list
#     # each output image is an array of complex numbers
#     output_image_list = []
#     for output_fft_row in output_fft_image_array:
#         output_image_row = []
#         for each_output_fft in output_fft_row:
#             output_image_row.append(np.fft.ifft2(each_output_fft))
#         output_image_list.append(output_image_row)
# 
#     # extract the magnitudes of the output images
#     mag_output_image_list = []
#     for each_image_row in output_image_list:
#         mag_output_image_row = []
#         for each_image in each_image_row:
#             mag_output_image_row.append(magnitude(each_image))
#         mag_output_image_list.append(mag_output_image_row)
# 
#     # plot the magnitudes of the output images in an n by n matrix shape
#     plot_2d_gray_multi(mag_output_image_list)


if __name__ == '__main__':
    task_1()
    task_2()
    task_2(DEFAULT_IMAGES[1])
    task_2(DEFAULT_IMAGES[3])
    task_3()